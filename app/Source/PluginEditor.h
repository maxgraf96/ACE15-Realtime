// ACE15 Realtime — WebView editor. Hosts a juce::WebBrowserComponent that loads
// the baked HTML/JS/CSS from BinaryData and bridges JS <-> native via JUCE 8's
// withNativeFunction / emitEvent. JS controls call into the processor; engine
// EVENTs (loaded/styled/playing/download_progress/error) are pushed to JS.
#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class ACE15Editor : public juce::AudioProcessorEditor,
                    public juce::DragAndDropContainer
{
public:
    explicit ACE15Editor(ACE15Processor&);
    ~ACE15Editor() override;

    void resized() override;

private:
    std::optional<juce::WebBrowserComponent::Resource> provide(const juce::String& url);
    void emitToJs(const juce::var& event);

    ACE15Processor& processor;
    juce::WebBrowserComponent webView;
    std::unique_ptr<juce::FileChooser> chooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ACE15Editor)
};
