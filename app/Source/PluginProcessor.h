// ACE15 Realtime — audio processor.
//
// Generator: no audio input. The audio callback drains PCM (48k) streamed from
// the Python cover sidecar over IPC and resamples to the host sample rate. The
// editor (WebView) drives it via the control methods, which forward CONTROL
// frames to the sidecar. Engine EVENTs are forwarded to the editor.
#pragma once

#include <JuceHeader.h>
#include <atomic>
#include "IpcClient.h"

class ACE15Processor : public juce::AudioProcessor
{
public:
    ACE15Processor();
    ~ACE15Processor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "ACE15 Realtime"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}
    void getStateInformation(juce::MemoryBlock&) override {}
    void setStateInformation(const void*, int) override {}

    // ---- editor-facing control API (message thread) ----
    void loadTrack(const juce::String& path, double seconds);
    void uploadAudio(const juce::String& base64, const juce::String& ext);
    void setStyle(const juce::String& tags, double denoise, double character);
    void setPrompt(const juce::String& tags);
    void enhance(const juce::String& tags);   // rewrite a short style into a rich caption (5Hz LM)
    void setDenoise(double v);
    void setCharacter(double v);
    void setMetas(bool sendBpm, bool sendKey, const juce::String& bpm, const juce::String& key);   // tempo/key flags + (edited) values
    void setEvolve(bool on);
    void setDcw(bool on);   // wavelet-domain per-step correction (DCW) on/off
    void seek(double fraction);   // jump playback to a fractional position (0..1)
    void reconfigure(int steps, double window);
    void setModel(const juce::String& m);   // "fast"(2B) / "quality"(XL); reloads current track
    void setInputGain(double db);   // trim feeding the model (<=0 dB); re-encodes the source
    void setMakeup(double db);      // make-up gain right before audio-out (-20..+20 dB)
    void setRealtimeInput(bool on); // (Phase A) stream the input bus to the engine (capture only)
    // Real-time live mode: start/stop generating a live accompaniment from the input bus.
    void startRealtime(const juce::String& tags, double denoise, double character,
                       const juce::String& bpm, const juce::String& key);
    void stopRealtime();
    void setStems(const juce::var& stems);   // live source separation: keep only these stems (e.g. ["drums"])
    void play();    // play / resume (keeps position)
    void pause();   // pause — keep position
    void stop();    // full stop — reset to the start
    void setBypass(bool on);   // A/B: hear the original source instead of the cover (instant)
    bool engineConnected() const { return ipc.isConnected(); }

    // Editor sets this to receive engine EVENTs (on the message thread).
    std::function<void(juce::var)> onEngineEvent;

private:
    void ensureSidecar();

    void sendLoad(juce::var load);

    IpcClient ipc;
    juce::ChildProcess sidecar;
    bool sidecarSpawned = false;
    juce::String selectedModel { "quality" };  // default XL; "fast"(2B) / "quality"(XL)
    bool sendBpm { true }, sendKey { true };   // inject detected tempo/key into prompt Metas
    juce::String metaBpm, metaKey;             // user-edited tempo/key (empty = use auto-detected)
    juce::var liveStems;                        // selected live stems (array; empty = full mix)
    bool dcwEnabled { false };              // DCW correction; off by default (runs hot in our regime)
    juce::var lastLoad;                      // last load message, for model-change reload

    double hostSampleRate = 48000.0;
    // 48k -> host-SR resampling (only when the device isn't 48k). A persistent
    // windowed-sinc interpolator per channel + a leftover buffer of un-consumed
    // 48k input, so phase + history stay continuous across blocks (linear interp
    // with a per-block phase reset aliased badly -> "metallic" on non-48k devices).
    juce::WindowedSincInterpolator resampler[2];
    juce::AudioBuffer<float> rsIn;       // staged 48k input: [leftover | freshly popped]
    int rsLeftover = 0;                  // un-consumed input samples held at the front of rsIn
    // Live INPUT host-SR -> 48k, resampled CONTINUOUSLY (phase-tracked) so the engine gets a
    // clean source (per-chunk resampling drifts/warbles). Plus a dry copy for input monitoring.
    juce::WindowedSincInterpolator inResamp[2];
    juce::AudioBuffer<float> inStage;    // staged host-SR input awaiting resample
    int inStageLen = 0;
    juce::AudioBuffer<float> in48;       // resampled 48k input to stream to the engine
    juce::AudioBuffer<float> monBuf;     // dry input (host SR) saved for the monitor mix
    bool playing = false;
    std::atomic<float> makeupLin { 1.0f };  // make-up gain (linear), applied just before output
    std::atomic<bool> captureInput { false }; // real-time mode: stream the live input bus to the engine

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ACE15Processor)
};
