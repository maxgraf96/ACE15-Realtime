// IPC client for the ACE15 cover sidecar.
//
// Speaks the framed TCP protocol from sidecar/server.py:
//   frame = [4-byte big-endian length][1-byte type][payload]
//     0x01 CONTROL  (we send, JSON)
//     0x02 AUDIO    (we receive, float32 interleaved 4ch @ 48k:
//                    [coverL,coverR, origL,origR])
//     0x03 EVENT    (we receive, JSON)
//     0x04 AUDIO-IN (we send, float32 interleaved stereo @ host SR — live input
//                    for real-time mode; the sidecar resamples to 48k)
//
// A background net thread receives frames; AUDIO is deinterleaved into a
// lock-free SPSC ring (AbstractFifo) drained by the audio callback; EVENTs are
// marshalled to the message thread. Controls are sent from the message thread.
//
// The engine sends both the cover AND the original source (frame-aligned), so
// `bypass` flips which pair the audio callback outputs — instant A/B, no regen,
// no drift (one ring, one read cursor).
#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>

class IpcClient
{
public:
    static constexpr int kStreamSampleRate = 48000;

    IpcClient();
    ~IpcClient();

    // Connect with retry (the sidecar may still be starting). Returns true once connected.
    bool connect(const juce::String& host, int port, int retryMs = 8000);
    void disconnect();
    bool isConnected() const { return connected.load(); }

    // Message-thread: send a CONTROL JSON line.
    void sendControl(const juce::var& json);

    // Audio-thread: push captured INPUT (host-SR, planar) into a lock-free FIFO that
    // a sender thread streams to the engine as 0x04 frames (never blocks the RT thread).
    void pushAudioIn(const float* const* in, int numCh, int n);

    // Audio-thread: pull up to n frames into planar out[ch][..]; returns frames provided.
    // Outputs the cover pair, or the original-source pair when bypass is set.
    int popAudio(float* const* out, int numCh, int n);
    int framesAvailable() const { return fifo.getNumReady(); }

    // A/B: false = hear the cover, true = hear the original source. Instant (the
    // engine streams both, frame-aligned; this just picks which pair to output).
    void setBypass(bool b) { bypass.store(b); }

    // Stop housekeeping: discard everything buffered (audio thread does it on the
    // next callback) so playback silences immediately and nothing stale carries to
    // the next play. setStreamActive(false) makes the net thread DROP any audio
    // still in flight over the socket; re-enable on play.
    void flushRing() { flushPending.store(true); }
    void setStreamActive(bool a) { streamActive.store(a); }

    // EVENT callback (invoked on the message thread).
    std::function<void(juce::var)> onEvent;

private:
    void netLoop();
    void inLoop();   // input-sender thread: drains inFifo -> 0x04 AUDIO-IN frames
    bool recvExact(void* dst, int n);
    void pushAudioInterleaved(const float* inter, int frames, int channels);
    void writeFrame(unsigned char type, const void* data, int bytes);  // socket write under writeMutex

    juce::StreamingSocket socket;
    std::thread thread;
    std::thread inThread;
    std::mutex writeMutex;                     // serialize socket writes (control + audio-in)
    std::atomic<bool> running { false };
    std::atomic<bool> connected { false };
    std::atomic<bool> bypass { false };       // A/B: output the original-source pair instead of the cover
    std::atomic<bool> flushPending { false }; // audio thread discards the ring on the next callback
    std::atomic<bool> streamActive { true };  // false => net thread drops incoming audio (post-stop in-flight)

    static constexpr int kStreamChannels = 4; // wire: [coverL,coverR, origL,origR]
    static constexpr int kRingFrames = kStreamSampleRate * 8; // 8 s jitter buffer
    juce::AbstractFifo fifo { kRingFrames };
    juce::AudioBuffer<float> ring; // [4, kRingFrames] planar: cover pair + original pair

    // Captured INPUT staged at host SR (planar stereo); the inLoop thread interleaves
    // + ships it as 0x04. The sidecar resamples host->48k.
    static constexpr int kInRingFrames = 192000 * 4; // 4 s at up to 192 kHz
    juce::AbstractFifo inFifo { kInRingFrames };
    juce::AudioBuffer<float> inRing;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(IpcClient)
};
