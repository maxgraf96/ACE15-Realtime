#include "PluginProcessor.h"
#include "PluginEditor.h"

// Soft-clip safety limiter. The cover output regularly peaks > 1.0 (dense
// styles / high Amount), and streaming raw PCM to the device hard-clips ->
// harsh "metallic" distortion. Transparent below 0.7, soft-saturates above
// (max -> 1.0), so loudness/transients survive but nothing hard-clips.
static inline float softClip(float x) noexcept
{
    constexpr float t = 0.92f;   // transparent below; only tames true overshoots (no transient crunch)
    const float a = std::abs(x);
    if (a <= t) return x;
    return (x < 0.0f ? -1.0f : 1.0f) * (t + (1.0f - t) * std::tanh((a - t) / (1.0f - t)));
}

ACE15Processor::ACE15Processor()
    : juce::AudioProcessor(BusesProperties().withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
    ipc.onEvent = [this](juce::var v)
    {
        // The engine's stats "progress" is the position it has SENT; what's actually
        // heard lags by the audio still sitting in our jitter ring. Subtract that so
        // the playhead matches the speakers.
        if (v.getProperty("event", {}).toString() == "stats")
        {
            if (auto* o = v.getDynamicObject())
            {
                const double dur = (double) o->getProperty("duration_s");
                if (dur > 0.0)
                {
                    const double lag = ipc.framesAvailable() / (double) IpcClient::kStreamSampleRate;
                    double p = (double) o->getProperty("progress") - lag / dur;
                    if (p < 0.0) p = 0.0;   // near start/loop, heard pos not caught up — clamp (don't wrap to end)
                    o->setProperty("progress", p);
                }
            }
        }
        if (onEngineEvent) onEngineEvent(v);
    };
    ensureSidecar();
    // Connect in the background (sidecar may still be starting).
    juce::Thread::launch([this] { ipc.connect("127.0.0.1", 8765, 20000); });
}

ACE15Processor::~ACE15Processor()
{
    ipc.disconnect();
    if (sidecarSpawned && sidecar.isRunning())
    {
        ipc.sendControl(juce::var()); // best-effort; sidecar exits on socket close
        sidecar.kill();
    }
}

void ACE15Processor::ensureSidecar()
{
    // Dev: ACE15_SPAWN_SIDECAR=1 spawns the venv sidecar. Otherwise assume one
    // is already running (manual `python sidecar/server.py`). The bundled app
    // will spawn the embedded interpreter here.
    if (juce::SystemStats::getEnvironmentVariable("ACE15_SPAWN_SIDECAR", "0") != "1")
        return;
    const auto py = juce::SystemStats::getEnvironmentVariable("ACE15_PYTHON", "");
    const auto script = juce::SystemStats::getEnvironmentVariable("ACE15_SIDECAR", "");
    if (py.isEmpty() || script.isEmpty()) return;
    juce::StringArray cmd { py, script };
    sidecarSpawned = sidecar.start(cmd);
}

void ACE15Processor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    hostSampleRate = sampleRate;
    const double ratio = (double) IpcClient::kStreamSampleRate / sampleRate;
    resampleIn.setSize(2, (int) std::ceil(samplesPerBlock * ratio) + 8);
}

void ACE15Processor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    buffer.clear();
    const int numOut = buffer.getNumSamples();
    const int numCh = juce::jmin(2, buffer.getNumChannels());
    const double ratio = (double) IpcClient::kStreamSampleRate / hostSampleRate;

    if (std::abs(ratio - 1.0) < 1e-6)
    {
        float* outs[2] = { buffer.getWritePointer(0), numCh > 1 ? buffer.getWritePointer(1) : buffer.getWritePointer(0) };
        ipc.popAudio(outs, numCh, numOut);
        for (int ch = 0; ch < numCh; ++ch)
        {
            float* d = buffer.getWritePointer(ch);
            for (int i = 0; i < numOut; ++i) d[i] = softClip(d[i]);
        }
        return;
    }

    // Pull numIn @48k, linear-resample to numOut @ host SR (v1; Lagrange later).
    const int numIn = juce::jmin(resampleIn.getNumSamples(), (int) std::lround(numOut * ratio));
    float* ins[2] = { resampleIn.getWritePointer(0), resampleIn.getWritePointer(1) };
    ipc.popAudio(ins, 2, numIn);
    for (int ch = 0; ch < numCh; ++ch)
    {
        const float* in = resampleIn.getReadPointer(juce::jmin(ch, 1));
        float* out = buffer.getWritePointer(ch);
        for (int i = 0; i < numOut; ++i)
        {
            const double pos = i * ratio;
            const int i0 = (int) pos;
            const int i1 = juce::jmin(i0 + 1, numIn - 1);
            const float f = (float) (pos - i0);
            out[i] = softClip(in[juce::jlimit(0, numIn - 1, i0)] * (1.0f - f) + in[i1] * f);
        }
    }
}

// ---- control API ----
static juce::var makeMsg(const juce::String& cmd)
{
    auto* o = new juce::DynamicObject();
    o->setProperty("cmd", cmd);
    return juce::var(o);
}

void ACE15Processor::sendLoad(juce::var load)
{
    load.getDynamicObject()->setProperty("model", selectedModel);
    lastLoad = load;            // remembered so a model change can reload the same track
    ipc.sendControl(load);
}

void ACE15Processor::loadTrack(const juce::String& path, double seconds)
{
    auto m = makeMsg("load");
    m.getDynamicObject()->setProperty("path", path);
    if (seconds > 0) m.getDynamicObject()->setProperty("seconds", seconds);
    sendLoad(m);
}

void ACE15Processor::uploadAudio(const juce::String& base64, const juce::String& ext)
{
    auto m = makeMsg("load");
    m.getDynamicObject()->setProperty("file_b64", base64);
    m.getDynamicObject()->setProperty("ext", ext);
    sendLoad(m);
}

void ACE15Processor::setModel(const juce::String& mdl)
{
    if (mdl == selectedModel) return;
    selectedModel = mdl;
    if (lastLoad.isObject())   // reload the current track with the new model
        sendLoad(lastLoad);
}

void ACE15Processor::setStyle(const juce::String& tags, double denoise, double character)
{
    auto m = makeMsg("style");
    m.getDynamicObject()->setProperty("tags", tags);
    m.getDynamicObject()->setProperty("denoise", denoise);
    m.getDynamicObject()->setProperty("character", character);
    m.getDynamicObject()->setProperty("send_bpm", sendBpm);
    m.getDynamicObject()->setProperty("send_key", sendKey);
    m.getDynamicObject()->setProperty("dcw", dcwEnabled);   // start the handle in the right DCW state
    ipc.sendControl(m);
}

void ACE15Processor::setMetas(bool bpmOn, bool keyOn)
{
    sendBpm = bpmOn; sendKey = keyOn;
    auto m = makeMsg("metas");
    m.getDynamicObject()->setProperty("send_bpm", bpmOn);
    m.getDynamicObject()->setProperty("send_key", keyOn);
    ipc.sendControl(m);
}

void ACE15Processor::setPrompt(const juce::String& tags)
{
    auto m = makeMsg("prompt");
    m.getDynamicObject()->setProperty("tags", tags);
    ipc.sendControl(m);
}

void ACE15Processor::setDenoise(double v)
{
    auto m = makeMsg("denoise");
    m.getDynamicObject()->setProperty("value", v);
    ipc.sendControl(m);
}

void ACE15Processor::setCharacter(double v)
{
    auto m = makeMsg("character");
    m.getDynamicObject()->setProperty("value", v);
    ipc.sendControl(m);
}

void ACE15Processor::setEvolve(bool on)
{
    auto m = makeMsg("evolve");
    m.getDynamicObject()->setProperty("value", on);
    ipc.sendControl(m);
}

void ACE15Processor::setDcw(bool on)
{
    dcwEnabled = on;   // remembered so setStyle/reload re-applies it
    auto m = makeMsg("dcw");
    m.getDynamicObject()->setProperty("value", on);
    ipc.sendControl(m);
}

void ACE15Processor::seek(double fraction)
{
    auto m = makeMsg("seek");
    m.getDynamicObject()->setProperty("value", fraction);
    ipc.sendControl(m);
}

void ACE15Processor::reconfigure(int steps, double window)
{
    auto m = makeMsg("reconfigure");
    m.getDynamicObject()->setProperty("steps", steps);
    m.getDynamicObject()->setProperty("window", window);
    ipc.sendControl(m);
}

void ACE15Processor::play() { ipc.sendControl(makeMsg("play")); playing = true; }
void ACE15Processor::stop() { ipc.sendControl(makeMsg("stop")); playing = false; }

juce::AudioProcessorEditor* ACE15Processor::createEditor() { return new ACE15Editor(*this); }

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new ACE15Processor(); }
