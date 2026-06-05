#include "PluginEditor.h"
#include "BinaryData.h"

using Resource = juce::WebBrowserComponent::Resource;
static constexpr auto kOrigin = "juce://ace15.local";

static const char* mimeFor(const juce::String& url)
{
    if (url.endsWith(".html")) return "text/html";
    if (url.endsWith(".js"))   return "text/javascript";
    if (url.endsWith(".css"))  return "text/css";
    return "application/octet-stream";
}

ACE15Editor::ACE15Editor(ACE15Processor& p)
    : juce::AudioProcessorEditor(p), processor(p),
      webView(juce::WebBrowserComponent::Options{}
          .withNativeIntegrationEnabled()
          .withResourceProvider([this](const auto& url) { return provide(url); },
                                juce::URL(kOrigin).getOrigin())
          .withNativeFunction("openFile", [this](auto, auto completion)
          {
              chooser = std::make_unique<juce::FileChooser>("Choose a source track",
                            juce::File{}, "*.wav;*.flac;*.mp3;*.aif;*.aiff");
              chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                  [this, completion](const juce::FileChooser& fc)
                  {
                      auto f = fc.getResult();
                      if (f.existsAsFile())
                          processor.loadTrack(f.getFullPathName(), 0.0);
                      completion(f.getFullPathName());
                  });
          })
          .withNativeFunction("uploadAudio", [this](auto args, auto completion)
          {
              processor.uploadAudio(args[0].toString(), args.size() > 1 ? args[1].toString() : ".wav");
              completion({});
          })
          .withNativeFunction("setStyle", [this](auto args, auto completion)
          {
              processor.setStyle(args[0].toString(), (double) args[1], args.size() > 2 ? (double) args[2] : 0.0);
              completion({});
          })
          .withNativeFunction("enhance", [this](auto args, auto completion)
          {
              processor.enhance(args[0].toString()); completion({});
          })
          .withNativeFunction("setPrompt", [this](auto args, auto completion)
          {
              processor.setPrompt(args[0].toString()); completion({});
          })
          .withNativeFunction("setDenoise", [this](auto args, auto completion)
          {
              processor.setDenoise((double) args[0]); completion({});
          })
          .withNativeFunction("setCharacter", [this](auto args, auto completion)
          {
              processor.setCharacter((double) args[0]); completion({});
          })
          .withNativeFunction("setEvolve", [this](auto args, auto completion)
          {
              processor.setEvolve((bool) args[0]); completion({});
          })
          .withNativeFunction("setDcw", [this](auto args, auto completion)
          {
              processor.setDcw((bool) args[0]); completion({});
          })
          .withNativeFunction("setSepBypass", [this](auto args, auto completion)
          {
              processor.setSepBypass((bool) args[0]); completion({});
          })
          .withNativeFunction("seek", [this](auto args, auto completion)
          {
              processor.seek((double) args[0]); completion({});
          })
          .withNativeFunction("reconfigure", [this](auto args, auto completion)
          {
              processor.reconfigure((int) args[0], (double) args[1]); completion({});
          })
          .withNativeFunction("setModel", [this](auto args, auto completion)
          {
              processor.setModel(args[0].toString()); completion({});
          })
          .withNativeFunction("setLiveMode", [this](auto args, auto completion)
          {
              processor.setLiveMode((bool) args[0]); completion({});
          })
          .withNativeFunction("setInputGain", [this](auto args, auto completion)
          {
              processor.setInputGain((double) args[0]); completion({});
          })
          .withNativeFunction("setMakeup", [this](auto args, auto completion)
          {
              processor.setMakeup((double) args[0]); completion({});
          })
          .withNativeFunction("setRealtimeInput", [this](auto args, auto completion)
          {
              processor.setRealtimeInput((bool) args[0]); completion({});
          })
          .withNativeFunction("startRealtime", [this](auto args, auto completion)
          {
              processor.startRealtime(args[0].toString(), (double) args[1], (double) args[2],
                                      args.size() > 3 ? args[3].toString() : juce::String(),
                                      args.size() > 4 ? args[4].toString() : juce::String(),
                                      args.size() > 5 ? (double) args[5] : 0.0,
                                      args.size() > 6 ? (double) args[6] : 0.0);
              completion({});
          })
          .withNativeFunction("stopRealtime", [this](auto, auto completion) { processor.stopRealtime(); completion({}); })
          .withNativeFunction("setStems", [this](auto args, auto completion) { processor.setStems(args[0]); completion({}); })
          .withNativeFunction("setLoopBars", [this](auto args, auto completion) { processor.setLoopBars((double) args[0]); completion({}); })
          .withNativeFunction("setLoopLead", [this](auto args, auto completion) { processor.setLoopLead((double) args[0]); completion({}); })
          .withNativeFunction("setMetas", [this](auto args, auto completion)
          {
              processor.setMetas((bool) args[0], (bool) args[1],
                                 args.size() > 2 ? args[2].toString() : juce::String(),
                                 args.size() > 3 ? args[3].toString() : juce::String());
              completion({});
          })
          .withNativeFunction("play", [this](auto, auto completion) { processor.play(); completion({}); })
          .withNativeFunction("pause", [this](auto, auto completion) { processor.pause(); completion({}); })
          .withNativeFunction("stop", [this](auto, auto completion) { processor.stop(); completion({}); })
          .withNativeFunction("setBypass", [this](auto args, auto completion) { processor.setBypass((bool) args[0]); completion({}); }))
{
    processor.onEngineEvent = [this](juce::var v) { emitToJs(v); };
    addAndMakeVisible(webView);
    webView.goToURL(juce::String(kOrigin) + "/index.html");
    setResizable(true, true);
    setSize(960, 720);
}

ACE15Editor::~ACE15Editor() { processor.onEngineEvent = nullptr; }

void ACE15Editor::resized() { webView.setBounds(getLocalBounds()); }

void ACE15Editor::emitToJs(const juce::var& event)
{
    webView.emitEventIfBrowserIsVisible("engineEvent", event);
}

std::optional<Resource> ACE15Editor::ACE15Editor::provide(const juce::String& url)
{
    juce::String name = url.fromLastOccurrenceOf("/", false, false);
    if (name.isEmpty() || name == "index.html") name = "index.html";

    int size = 0;
    juce::String resName = name.replaceCharacter('.', '_').replaceCharacter('-', '_');
    if (const char* data = BinaryData::getNamedResource(resName.toRawUTF8(), size))
    {
        std::vector<std::byte> bytes((size_t) size);
        std::memcpy(bytes.data(), data, (size_t) size);
        return Resource{ std::move(bytes), juce::String(mimeFor(name)) };
    }
    return std::nullopt;
}
