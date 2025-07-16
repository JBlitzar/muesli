import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15

ApplicationWindow {
    id: mainWindow
    visible: true
    width: 1024
    height: 768
    title: "Muesli - Voice Transcription & Summarization"
    
    // Properties for binding with Python
    property string transcriptText: ""
    property string summaryText: ""
    property string statusMessage: "Ready"
    property real transcriptionProgress: 0.0
    property bool isRecording: false
    property bool isTranscribing: false
    property bool isSummarizing: false
    property bool hasTranscript: false
    property bool hasSummary: false       // kept for compatibility (may be unused)
    // NEW: combined markdown/text shown in a single viewer
    property string combinedContent: ""
    
    // Color scheme
    property color primaryColor: "#2c3e50"
    property color accentColor: "#3498db"
    property color textColor: "#333333"
    property color lightTextColor: "#7f8c8d"
    property color backgroundColor: "#fcfbf6"
    property color cardColor: "#ffffff"
    
    // Font settings
    property string fontFamily: "Inter, Helvetica Neue, Arial, sans-serif"
    property int baseFontSize: 18
    /* Simple ASCII spinner setup */
    property var  spinnerChars: ["-", "\\\\", "|", "/"]
    property int  spinnerIdx: 0
    property bool spinnerActive: isTranscribing || isSummarizing
    property string spinnerChar: spinnerChars[spinnerIdx]

    /* Timer to advance spinner when active */
    Timer {
        id: spinnerTimer
        interval: 150
        running: spinnerActive
        repeat: true
        onTriggered: spinnerIdx = (spinnerIdx + 1) % spinnerChars.length
    }
    
    // Main content
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 12
        
        // Toolbar
        Rectangle {
            Layout.fillWidth: true
            height: 50
            color: primaryColor
            radius: 4
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 8
                spacing: 8
                
                // Logo/Title
                Label {
                    text: "Muesli"
                    color: "white"
                    font.pixelSize: baseFontSize
                    font.bold: true
                    font.family: fontFamily
                }
                
                // Spacer
                Item { Layout.fillWidth: true }
                
                // Open file button
                Button {
                    id: openFileButton
                    text: "Open File"
                    icon.name: "document-open"
                    
                    onClicked: {
                        // Will be connected to Python
                        openFileRequested()
                    }
                }
                
                // Record button
                Button {
                    id: recordButton
                    text: isRecording ? "Stop Recording" : "Record"
                    icon.name: isRecording ? "media-playback-stop" : "media-record"
                    
                    onClicked: {
                        // Will be connected to Python
                        recordRequested()
                    }
                }
                
                // Transcribe button
                Button {
                    id: transcribeButton
                    text: "Transcribe"
                    icon.name: "document-edit"
                    enabled: !isTranscribing
                    
                    onClicked: {
                        // Will be connected to Python
                        transcribeRequested()
                    }
                }
                
                // Summarize button
                Button {
                    id: summarizeButton
                    text: "Summarize"
                    icon.name: "view-list-text"
                    enabled: hasTranscript && !isSummarizing
                    
                    onClicked: {
                        // Will be connected to Python
                        summarizeRequested()
                    }
                }
            }
        }
        
        // Main content area ‒ single combined view
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: cardColor
            radius: 4
            
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 12
                
                // Header
                RowLayout {
                    Layout.fillWidth: true
                    
                    Label {
                        text: "Summary & Transcript"
                        font.pixelSize: baseFontSize
                        font.bold: true
                        font.family: fontFamily
                        color: textColor
                    }
                    
                    Label {
                        text: (isTranscribing ? "Transcribing..." :
                               (isSummarizing ? "Summarizing..." :
                                (hasTranscript ? "Content ready" : "No content")))
                        color: lightTextColor
                        font.family: fontFamily
                    }
                    
                    Item { Layout.fillWidth: true }
                    
                    // Text-based loading spinner
                    RowLayout {
                        visible: spinnerActive
                        spacing: 4
                        Label {
                            text: "Loading"
                            color: lightTextColor
                            font.family: fontFamily
                            font.pixelSize: baseFontSize
                        }
                        Label {
                            text: spinnerChar
                            color: accentColor
                            font.family: fontFamily
                            font.pixelSize: baseFontSize
                        }
                    }
                }
                
                // Combined content
                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    
                    TextArea {
                        id: combinedTextArea
                        text: combinedContent
                        wrapMode: TextEdit.Wrap
                        width: parent.width
                        readOnly: true
                        font.pixelSize: baseFontSize
                        font.family: fontFamily
                        lineHeight: 1.5
                        leftPadding: 16
                        rightPadding: 16
                        topPadding: 8
                        bottomPadding: 8
                        
                        // Set a preferred width for the text to enforce ~65 character limit
                        // This is approximate as character width varies with proportional fonts
                        Layout.preferredWidth: baseFontSize * 40
                        
                        background: Rectangle { 
                            color: "transparent"
                            border.width: 1
                            border.color: "#e0e0e0"
                            radius: 4
                        }
                        
                        Label {
                            anchors.fill: parent
                            text: "Summary and transcript will appear here"
                            color: lightTextColor
                            font.family: fontFamily
                            font.pixelSize: baseFontSize
                            visible: combinedContent.length === 0
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                }
            }
        }
        
        // Status bar
        Rectangle {
            Layout.fillWidth: true
            height: 36
            color: primaryColor
            radius: 4
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 8
                spacing: 8
                
                Label {
                    text: statusMessage
                    color: "white"
                    font.pixelSize: 14
                    font.family: fontFamily
                    Layout.fillWidth: true
                }
                
                // Progress bar (only visible during operations)
                ProgressBar {
                    id: progressBar
                    value: transcriptionProgress
                    visible: isTranscribing
                    width: 150
                }
                
                // Alternative loading indicator for status bar
                RowLayout {
                    visible: spinnerActive
                    spacing: 2
                    Label {
                        text: "Loading"
                        color: "white"
                        font.family: fontFamily
                        font.pixelSize: 14
                    }
                    Label {
                        text: spinnerChar
                        color: accentColor
                        font.family: fontFamily
                        font.pixelSize: 14
                    }
                }
            }
        }
    }
    
    // JavaScript function to wrap text at approximately 65 characters
    // This is used when displaying the content
    function wrapText(text, width) {
        if (!text) return "";
        
        const lines = text.split('\n');
        let result = [];
        
        for (let i = 0; i < lines.length; i++) {
            let line = lines[i];
            if (line.length <= width || line.trim() === "") {
                result.push(line);
            } else {
                // Simple word wrapping - not perfect but helps approximate the 65 char limit
                let currentLine = "";
                const words = line.split(' ');
                
                for (let j = 0; j < words.length; j++) {
                    const word = words[j];
                    if ((currentLine + " " + word).length <= width) {
                        currentLine += (currentLine ? " " : "") + word;
                    } else {
                        result.push(currentLine);
                        currentLine = word;
                    }
                }
                
                if (currentLine) {
                    result.push(currentLine);
                }
            }
        }
        
        return result.join('\n');
    }
    
    // Signal handlers for Python integration
    signal openFileRequested()
    signal recordRequested()
    signal transcribeRequested()
    signal summarizeRequested()
    signal saveTranscriptRequested()
    signal saveSummaryRequested()
    
    // Function to update transcript text from Python
    function updateTranscript(text) {
        // Back-compat: still update transcriptText
        transcriptText = text;
        // Wrap text to approximate 65 characters per line
        const wrappedText = wrapText(text, 65);
        
        // If we already have a summary, prepend it with separator
        if (summaryText.length > 0) {
            combinedContent = summaryText + "\n\n---\n\n" + wrappedText;
        } else {
            combinedContent = wrappedText;
        }
        hasTranscript = text.length > 0;
    }
    
    // Function to update summary text from Python
    function updateSummary(text) {
        summaryText = text;
        hasSummary = text.length > 0;
        
        // Get wrapped version of transcript
        const wrappedTranscript = wrapText(transcriptText, 65);
        
        // Prepend summary, followed by separator and existing transcript
        if (transcriptText.length > 0) {
            combinedContent = text + "\n\n---\n\n" + wrappedTranscript;
        } else {
            combinedContent = text;
        }
    }
    
    // Function to update status from Python
    function updateStatus(message, progress) {
        statusMessage = message;
        if (progress !== undefined) {
            transcriptionProgress = progress;
        }
    }
    
    // Function to set recording state from Python
    function setRecordingState(recording) {
        isRecording = recording;
    }
    
    // Function to set transcribing state from Python
    function setTranscribingState(transcribing) {
        isTranscribing = transcribing;
    }
    
    // Function to set summarizing state from Python
    function setSummarizingState(summarizing) {
        isSummarizing = summarizing;
    }
}
