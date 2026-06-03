#include "IpcClient.h"

IpcClient::IpcClient()
{
    ring.setSize(kStreamChannels, kRingFrames);
    ring.clear();
}

IpcClient::~IpcClient() { disconnect(); }

bool IpcClient::connect(const juce::String& host, int port, int retryMs)
{
    const auto deadline = juce::Time::getMillisecondCounter() + (juce::uint32) retryMs;
    while (juce::Time::getMillisecondCounter() < deadline)
    {
        if (socket.connect(host, port, 300))
        {
            connected = true;
            running = true;
            thread = std::thread([this] { netLoop(); });
            return true;
        }
        juce::Thread::sleep(150);
    }
    return false;
}

void IpcClient::disconnect()
{
    running = false;
    socket.close();
    if (thread.joinable())
        thread.join();
    connected = false;
}

bool IpcClient::recvExact(void* dst, int n)
{
    auto* p = static_cast<char*>(dst);
    int got = 0;
    while (got < n && running)
    {
        const int r = socket.read(p + got, n - got, true);
        if (r <= 0)
            return false;
        got += r;
    }
    return got == n;
}

void IpcClient::netLoop()
{
    while (running)
    {
        unsigned char hdr[5];
        if (! recvExact(hdr, 5))
            break;
        const juce::uint32 len = ((juce::uint32) hdr[0] << 24) | ((juce::uint32) hdr[1] << 16)
                               | ((juce::uint32) hdr[2] << 8) | (juce::uint32) hdr[3];
        const unsigned char type = hdr[4];

        juce::HeapBlock<char> payload((size_t) len + 1);
        if (len > 0 && ! recvExact(payload.get(), (int) len))
            break;

        if (type == 0x02) // AUDIO: float32 interleaved 4ch (cover pair + original pair)
        {
            if (streamActive.load()) // dropped while stopped, so post-stop in-flight audio can't leak into the next play
            {
                const int frames = (int) (len / (sizeof(float) * kStreamChannels));
                pushAudioInterleaved(reinterpret_cast<const float*>(payload.get()), frames, kStreamChannels);
            }
        }
        else if (type == 0x03) // EVENT: JSON
        {
            juce::String js(payload.get(), (size_t) len);
            auto parsed = juce::JSON::parse(js);
            juce::MessageManager::callAsync([this, parsed]
            {
                if (onEvent) onEvent(parsed);
            });
        }
    }
    connected = false;
}

void IpcClient::pushAudioInterleaved(const float* inter, int frames, int channels)
{
    int start1, size1, start2, size2;
    fifo.prepareToWrite(frames, start1, size1, start2, size2);
    auto deinterleave = [&](int dstStart, int n, int srcOffset)
    {
        for (int ch = 0; ch < channels; ++ch)
        {
            float* dst = ring.getWritePointer(ch) + dstStart;
            for (int i = 0; i < n; ++i)
                dst[i] = inter[(size_t) (srcOffset + i) * channels + ch];
        }
    };
    if (size1 > 0) deinterleave(start1, size1, 0);
    if (size2 > 0) deinterleave(start2, size2, size1);
    fifo.finishedWrite(size1 + size2);
}

int IpcClient::popAudio(float* const* out, int numCh, int n)
{
    if (flushPending.exchange(false))            // stop: drop everything buffered (this thread owns the read cursor)
        fifo.finishedRead(fifo.getNumReady());

    int start1, size1, start2, size2;
    fifo.prepareToRead(n, start1, size1, start2, size2);
    const int got = size1 + size2;
    const int pair = bypass.load() ? 2 : 0;   // ring ch 0/1 = cover, 2/3 = original source
    for (int ch = 0; ch < numCh; ++ch)
    {
        const int src = pair + juce::jmin(ch, 1); // mono-out fallback uses the pair's left
        float* d = out[ch];
        if (size1 > 0) juce::FloatVectorOperations::copy(d, ring.getReadPointer(src) + start1, size1);
        if (size2 > 0) juce::FloatVectorOperations::copy(d + size1, ring.getReadPointer(src) + start2, size2);
        for (int i = got; i < n; ++i) d[i] = 0.0f; // underrun -> silence
    }
    fifo.finishedRead(got);
    return got;
}

void IpcClient::sendControl(const juce::var& json)
{
    if (! connected) return;
    const juce::String s = juce::JSON::toString(json, true);
    const auto utf8 = s.toRawUTF8();
    const juce::uint32 len = (juce::uint32) s.getNumBytesAsUTF8();
    unsigned char hdr[5] = {
        (unsigned char) ((len >> 24) & 0xff), (unsigned char) ((len >> 16) & 0xff),
        (unsigned char) ((len >> 8) & 0xff),  (unsigned char) (len & 0xff), 0x01 };
    socket.write(hdr, 5);
    socket.write(utf8, (int) len);
}
