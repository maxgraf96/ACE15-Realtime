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
    rsIn.setSize(2, (int) std::ceil(samplesPerBlock * ratio) + 16);
    rsLeftover = 0;
    resampler[0].reset();
    resampler[1].reset();
    if (std::abs(ratio - 1.0) >= 1e-6)
        juce::Logger::writeToLog("[ACE15] output device SR=" + juce::String(sampleRate)
            + " -> resampling 48000 (ratio " + juce::String(ratio, 4) + "); 48k avoids it entirely");
}

void ACE15Processor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    buffer.clear();
    const int numOut = buffer.getNumSamples();
    const int numCh = juce::jmin(2, buffer.getNumChannels());
    const double ratio = (double) IpcClient::kStreamSampleRate / hostSampleRate;
    const float mk = makeupLin.load();   // make-up gain (linear), applied just before soft-clip out

    if (std::abs(ratio - 1.0) < 1e-6)
    {
        float* outs[2] = { buffer.getWritePointer(0), numCh > 1 ? buffer.getWritePointer(1) : buffer.getWritePointer(0) };
        ipc.popAudio(outs, numCh, numOut);
        for (int ch = 0; ch < numCh; ++ch)
        {
            float* d = buffer.getWritePointer(ch);
            for (int i = 0; i < numOut; ++i) d[i] = softClip(d[i] * mk);
        }
        return;
    }

    // 48k -> host SR via a persistent windowed-sinc interpolator (clean stopband,
    // continuous phase + history across blocks). rsIn holds [leftover | fresh]; the
    // interpolator consumes from the front and we carry the unconsumed tail forward.
    const int cap = rsIn.getNumSamples();
    const int needIn = juce::jmin(cap, (int) std::ceil(numOut * ratio) + 2);
    const int popN = juce::jmax(0, needIn - rsLeftover);
    float* ins[2] = { rsIn.getWritePointer(0) + rsLeftover, rsIn.getWritePointer(1) + rsLeftover };
    ipc.popAudio(ins, 2, popN);                       // zero-fills on underrun
    const int avail = rsLeftover + popN;
    int used = avail;
    for (int ch = 0; ch < numCh; ++ch)
    {
        used = resampler[ch].process(ratio, rsIn.getReadPointer(juce::jmin(ch, 1)),
                                     buffer.getWritePointer(ch), numOut);
        float* d = buffer.getWritePointer(ch);
        for (int i = 0; i < numOut; ++i) d[i] = softClip(d[i] * mk);
    }
    rsLeftover = juce::jmax(0, avail - used);          // carry un-consumed input to next block
    if (rsLeftover > 0)
        for (int ch = 0; ch < 2; ++ch)
            juce::FloatVectorOperations::copy(rsIn.getWritePointer(ch),
                                              rsIn.getReadPointer(ch) + used, rsLeftover);
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

void ACE15Processor::setInputGain(double db)
{
    auto m = makeMsg("input_gain");
    m.getDynamicObject()->setProperty("value", db);
    ipc.sendControl(m);   // engine re-encodes the source (queued in the producer thread)
}

void ACE15Processor::setMakeup(double db)   // -20..+20 dB, applied in processBlock before soft-clip
{
    makeupLin.store((float) std::pow(10.0, db / 20.0));
}

void ACE15Processor::setStyle(const juce::String& tags, double denoise, double character)
{
    auto m = makeMsg("style");
    m.getDynamicObject()->setProperty("tags", tags);
    m.getDynamicObject()->setProperty("denoise", denoise);
    m.getDynamicObject()->setProperty("character", character);
    m.getDynamicObject()->setProperty("send_bpm", sendBpm);
    m.getDynamicObject()->setProperty("send_key", sendKey);
    if (metaBpm.isNotEmpty()) m.getDynamicObject()->setProperty("bpm", metaBpm);   // edited tempo (else engine uses detected)
    if (metaKey.isNotEmpty()) m.getDynamicObject()->setProperty("key", metaKey);   // edited key
    m.getDynamicObject()->setProperty("dcw", dcwEnabled);   // start the handle in the right DCW state
    ipc.sendControl(m);
}

void ACE15Processor::setMetas(bool bpmOn, bool keyOn, const juce::String& bpm, const juce::String& key)
{
    sendBpm = bpmOn; sendKey = keyOn;
    metaBpm = bpm.trim(); metaKey = key.trim();   // remembered so a re-style (style frame) keeps them
    auto m = makeMsg("metas");
    m.getDynamicObject()->setProperty("send_bpm", bpmOn);
    m.getDynamicObject()->setProperty("send_key", keyOn);
    if (metaBpm.isNotEmpty()) m.getDynamicObject()->setProperty("bpm", metaBpm);
    if (metaKey.isNotEmpty()) m.getDynamicObject()->setProperty("key", metaKey);
    ipc.sendControl(m);
}

void ACE15Processor::setPrompt(const juce::String& tags)
{
    auto m = makeMsg("prompt");
    m.getDynamicObject()->setProperty("tags", tags);
    ipc.sendControl(m);
}

void ACE15Processor::enhance(const juce::String& tags)
{
    auto m = makeMsg("enhance");
    m.getDynamicObject()->setProperty("tags", tags);
    ipc.sendControl(m);   // sidecar replies with an "enhanced" event (the rewritten caption)
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

void ACE15Processor::play()  { ipc.setStreamActive(true); ipc.sendControl(makeMsg("play")); playing = true; }  // play / resume
void ACE15Processor::pause() { ipc.sendControl(makeMsg("pause")); playing = false; }   // keep position (engine + C++ buffer)
// Full stop: silence NOW by flushing the C++ jitter ring + dropping any audio still in
// flight over the socket, so no pre-stop sound tails out or leaks into the next play.
// (The sidecar separately resets the engine to position 0.)
void ACE15Processor::stop()  { ipc.setStreamActive(false); ipc.flushRing(); ipc.sendControl(makeMsg("stop")); playing = false; }

// A/B: purely local — the engine already streams both cover and original, so this
// just flips which pair the audio callback outputs. No control frame, no latency.
void ACE15Processor::setBypass(bool on) { ipc.setBypass(on); }

juce::AudioProcessorEditor* ACE15Processor::createEditor() { return new ACE15Editor(*this); }

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new ACE15Processor(); }
