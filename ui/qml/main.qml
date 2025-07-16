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
    property color backgroundColor: "#f5f5f5"
    property color cardColor: "#ffffff"
    
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
                    font.pixelSize: 18
                    font.bold: true
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
                anchors.margins: 8
                spacing: 8

                // Header
                RowLayout {
                    Layout.fillWidth: true

                    Label {
                        text: "Summary & Transcript"
                        font.pixelSize: 16
                        font.bold: true
                        color: textColor
                    }

                    Label {
                        text: (isTranscribing ? "Transcribing..." :
                               (isSummarizing ? "Summarizing..." :
                                (hasTranscript ? "Content ready" : "No content")))
                        color: lightTextColor
                    }

                    Item { Layout.fillWidth: true }
                }

                // Combined content
                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    TextArea {
                        id: combinedTextArea
                        text: combinedContent
                        wrapMode: TextEdit.Wrap
                        readOnly: true
                        font.pixelSize: 14
                        background: Rectangle { color: "transparent" }

                        Label {
                            anchors.fill: parent
                            text: "Summary and transcript will appear here"
                            color: lightTextColor
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
            height: 30
            color: primaryColor
            radius: 4
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 4
                spacing: 8
                
                Label {
                    text: statusMessage
                    color: "white"
                    font.pixelSize: 12
                    Layout.fillWidth: true
                }
                
                // Progress bar (only visible during operations)
                ProgressBar {
                    id: progressBar
                    value: transcriptionProgress
                    visible: isTranscribing
                    width: 150
                }
            }
        }
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
        transcriptText = text
        // If we already have a summary, prepend it with separator
        if (summaryText.length > 0) {
            combinedContent = summaryText + "\n\n---\n\n" + text
        } else {
            combinedContent = text
        }
        hasTranscript = text.length > 0
    }
    
    // Function to update summary text from Python
    function updateSummary(text) {
        summaryText = text
        hasSummary = text.length > 0
        // Prepend summary, followed by separator and existing transcript
        if (transcriptText.length > 0) {
            combinedContent = text + "\n\n---\n\n" + transcriptText
        } else {
            combinedContent = text
        }
    }
    
    // Function to update status from Python
    function updateStatus(message, progress) {
        statusMessage = message
        if (progress !== undefined) {
            transcriptionProgress = progress
        }
    }
    
    // Function to set recording state from Python
    function setRecordingState(recording) {
        isRecording = recording
    }
    
    // Function to set transcribing state from Python
    function setTranscribingState(transcribing) {
        isTranscribing = transcribing
    }
    
    // Function to set summarizing state from Python
    function setSummarizingState(summarizing) {
        isSummarizing = summarizing
    }
}
