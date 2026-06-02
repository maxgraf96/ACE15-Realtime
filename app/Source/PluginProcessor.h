// ACE15 Realtime — audio processor.
//
// Generator: no audio input. The audio callback drains PCM (48k) streamed from
// the Python cover sidecar over IPC and resamples to the host sample rate. The
// editor (WebView) drives it via the control methods, which forward CONTROL
// frames to the sidecar. Engine EVENTs are forwarded to the editor.
#pragma once

#include <JuceHeader.h>
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
    void setDenoise(double v);
    void setCharacter(double v);
    void setMetas(bool sendBpm, bool sendKey);   // inject detected tempo/key into the prompt?
    void setEvolve(bool on);
    void setDcw(bool on);   // wavelet-domain per-step correction (DCW) on/off
    void seek(double fraction);   // jump playback to a fractional position (0..1)
    void reconfigure(int steps, double window);
    void setModel(const juce::String& m);   // "fast"(2B) / "quality"(XL); reloads current track
    void play();
    void stop();
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
    bool dcwEnabled { false };              // DCW correction; off by default (runs hot in our regime)
    juce::var lastLoad;                      // last load message, for model-change reload

    double hostSampleRate = 48000.0;
    juce::AudioBuffer<float> resampleIn; // staging at 48k
    bool playing = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ACE15Processor)
};
