import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15

ApplicationWindow {
    id: mainWindow
    visible: true
    width: 900
    height: 700
    title: "Muesli - Private Transcription"
    
    // Use system theme if available
    palette.window: Qt.styleHints.colorScheme === Qt.Dark ? "#2d2d2d" : "#f5f5f5"
    palette.windowText: Qt.styleHints.colorScheme === Qt.Dark ? "#ffffff" : "#000000"
    palette.base: Qt.styleHints.colorScheme === Qt.Dark ? "#1e1e1e" : "#ffffff"
    palette.text: Qt.styleHints.colorScheme === Qt.Dark ? "#ffffff" : "#000000"
    palette.button: Qt.styleHints.colorScheme === Qt.Dark ? "#3d3d3d" : "#e0e0e0"
    palette.buttonText: Qt.styleHints.colorScheme === Qt.Dark ? "#ffffff" : "#000000"
    palette.highlight: "#007acc"
    palette.highlightedText: "#ffffff"
    
    // Status bar at the bottom
    footer: Rectangle {
        height: 28
        color: palette.window
        border.color: Qt.darker(palette.window, 1.2)
        border.width: 1
        
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 10
            anchors.rightMargin: 10
            
            Label {
                text: transcriptModel.isRecording ? "Recording..." : "Ready"
                color: transcriptModel.isRecording ? "#e74c3c" : "#2ecc71"
                font.bold: transcriptModel.isRecording
            }
            
            Item { Layout.fillWidth: true }
            
            Label {
                text: summaryModel.isGenerating ? "Generating summary..." : ""
                color: "#f39c12"
                visible: summaryModel.isGenerating
                font.italic: true
            }
            
            Label {
                text: "Muesli v0.1.0"
                Layout.alignment: Qt.AlignRight
                font.pixelSize: 12
                color: Qt.darker(palette.windowText, 1.5)
            }
        }
    }
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12
        
        // Header with controls
        Rectangle {
            Layout.fillWidth: true
            height: 60
            color: Qt.lighter(palette.window, 1.05)
            radius: 8
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 16
                
                Button {
                    id: recordButton
                    Layout.preferredWidth: 140
                    Layout.preferredHeight: 36
                    
                    contentItem: Row {
                        spacing: 8
                        anchors.centerIn: parent
                        
                        Rectangle {
                            width: 12
                            height: 12
                            radius: 6
                            color: transcriptModel.isRecording ? "#e74c3c" : "#2ecc71"
                            anchors.verticalCenter: parent.verticalCenter
                            
                            // Pulsating animation when recording
                            SequentialAnimation on opacity {
                                running: transcriptModel.isRecording
                                loops: Animation.Infinite
                                NumberAnimation { to: 0.4; duration: 800; easing.type: Easing.InOutQuad }
                                NumberAnimation { to: 1.0; duration: 800; easing.type: Easing.InOutQuad }
                            }
                        }
                        
                        Text {
                            text: transcriptModel.isRecording ? "Stop Recording" : "Start Recording"
                            color: recordButton.palette.buttonText
                            font.pixelSize: 14
                            anchors.verticalCenter: parent.verticalCenter
                        }
                    }
                    
                    background: Rectangle {
                        radius: 4
                        color: recordButton.down ? Qt.darker(recordButton.palette.button, 1.1) : 
                               recordButton.hovered ? Qt.lighter(recordButton.palette.button, 1.1) : 
                               recordButton.palette.button
                        border.width: 1
                        border.color: Qt.darker(color, 1.3)
                    }
                    
                    onClicked: {
                        if (transcriptModel.isRecording) {
                            mainWindow.stopTranscription();
                        } else {
                            mainWindow.startTranscription();
                        }
                    }
                }
                
                Button {
                    id: summarizeButton
                    Layout.preferredWidth: 140
                    Layout.preferredHeight: 36
                    text: "Generate Summary"
                    enabled: !summaryModel.isGenerating && transcriptModel.text.length > 0
                    
                    background: Rectangle {
                        radius: 4
                        color: !summarizeButton.enabled ? Qt.darker(summarizeButton.palette.button, 1.2) :
                               summarizeButton.down ? Qt.darker(summarizeButton.palette.button, 1.1) : 
                               summarizeButton.hovered ? Qt.lighter(summarizeButton.palette.button, 1.1) : 
                               summarizeButton.palette.button
                        border.width: 1
                        border.color: Qt.darker(color, 1.3)
                        opacity: summarizeButton.enabled ? 1.0 : 0.6
                    }
                    
                    onClicked: mainWindow.generateSummary()
                }
                
                Item { Layout.fillWidth: true }
                
                ComboBox {
                    id: audioSourceCombo
                    Layout.preferredWidth: 180
                    model: ["Default Microphone", "System Audio", "File Input"]
                    currentIndex: 0
                    enabled: !transcriptModel.isRecording
                    
                    // In a real app, this would be connected to audio source selection
                }
            }
        }
        
        // Main content area with tabs
        TabBar {
            id: tabBar
            Layout.fillWidth: true
            
            TabButton {
                text: "Transcript"
                width: implicitWidth
            }
            TabButton {
                text: "Summary"
                width: implicitWidth
            }
        }
        
        StackLayout {
            currentIndex: tabBar.currentIndex
            Layout.fillWidth: true
            Layout.fillHeight: true
            
            // Transcript view
            Item {
                Rectangle {
                    anchors.fill: parent
                    color: palette.base
                    border.color: Qt.darker(palette.base, 1.1)
                    border.width: 1
                    radius: 4
                    
                    ScrollView {
                        id: transcriptScrollView
                        anchors.fill: parent
                        anchors.margins: 1
                        clip: true
                        
                        TextArea {
                            id: transcriptText
                            text: transcriptModel.text
                            readOnly: true
                            wrapMode: TextEdit.Wrap
                            selectByMouse: true
                            font.pixelSize: 14
                            padding: 16
                            
                            background: Rectangle {
                                color: "transparent"
                            }
                            
                            // Auto-scroll to bottom when text changes
                            onTextChanged: {
                                if (transcriptModel.isRecording) {
                                    transcriptScrollView.ScrollBar.vertical.position = 1.0
                                }
                            }
                            
                            // Placeholder text when empty
                            Rectangle {
                                anchors.fill: parent
                                color: "transparent"
                                visible: transcriptText.text === ""
                                
                                Text {
                                    anchors.centerIn: parent
                                    text: "Press 'Start Recording' to begin transcription"
                                    color: Qt.darker(palette.text, 1.6)
                                    font.pixelSize: 16
                                    font.italic: true
                                }
                            }
                        }
                    }
                }
            }
            
            // Summary view
            Item {
                Rectangle {
                    anchors.fill: parent
                    color: palette.base
                    border.color: Qt.darker(palette.base, 1.1)
                    border.width: 1
                    radius: 4
                    
                    ScrollView {
                        anchors.fill: parent
                        anchors.margins: 1
                        clip: true
                        
                        TextArea {
                            id: summaryText
                            text: summaryModel.text
                            readOnly: true
                            wrapMode: TextEdit.Wrap
                            selectByMouse: true
                            font.pixelSize: 14
                            padding: 16
                            
                            background: Rectangle {
                                color: "transparent"
                            }
                            
                            // Placeholder or loading state
                            Rectangle {
                                anchors.fill: parent
                                color: "transparent"
                                visible: summaryText.text === ""
                                
                                Column {
                                    anchors.centerIn: parent
                                    spacing: 12
                                    
                                    Text {
                                        anchors.horizontalCenter: parent.horizontalCenter
                                        text: summaryModel.isGenerating ? 
                                              "Generating summary..." : 
                                              "No summary generated yet"
                                        color: Qt.darker(palette.text, 1.6)
                                        font.pixelSize: 16
                                        font.italic: true
                                    }
                                    
                                    // Loading indicator
                                    BusyIndicator {
                                        anchors.horizontalCenter: parent.horizontalCenter
                                        running: summaryModel.isGenerating
                                        visible: summaryModel.isGenerating
                                    }
                                    
                                    Button {
                                        anchors.horizontalCenter: parent.horizontalCenter
                                        text: "Generate Summary"
                                        visible: !summaryModel.isGenerating && summaryText.text === "" && transcriptModel.text !== ""
                                        onClicked: mainWindow.generateSummary()
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Connections to handle signals from backend
    Connections {
        target: mainWindow
        
        function onTranscriptionError(errorMessage) {
            errorDialog.text = errorMessage
            errorDialog.open()
        }
        
        function onSummaryError(errorMessage) {
            errorDialog.text = errorMessage
            errorDialog.open()
        }
    }
    
    // Error dialog
    Dialog {
        id: errorDialog
        title: "Error"
        standardButtons: Dialog.Ok
        modal: true
        anchors.centerIn: parent
        width: 400
        
        property string text: ""
        
        contentItem: Label {
            text: errorDialog.text
            wrapMode: Text.Wrap
        }
    }
}
