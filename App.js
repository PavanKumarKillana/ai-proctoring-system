import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, View, Text, ActivityIndicator, SafeAreaView, Platform } from 'react-native';
import { WebView } from 'react-native-webview';
import { CameraView, useCameraPermissions } from 'expo-camera';

// Windows Mobile Hotspot IP — fixed forever, works at any location!
const SERVER_URL = 'http://192.168.137.1:7860';

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef(null);
  const webviewRef = useRef(null);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    if (!permission?.granted && permission?.canAskAgain) {
      requestPermission();
    }
  }, [permission]);

  // Inject JavaScript to intercept the exam logic
  // DELETED: Handled strictly by exam.html itself to fix WebView asynchronous initialization order bugs!
  const injectedScript = `true;`;

  const onMessage = async (event) => {
    try {
      const data = JSON.parse(event.nativeEvent.data);

      if (data.action === 'log') {
        console.log("[WebView Log]:", data.message);
        return;
      }

      if (data.action === 'take_picture' && cameraRef.current && !isProcessing) {
        setIsProcessing(true);

        // Take a highly compressed picture using the Native Expo Camera API
        // Added shutterSound: false to stop the clicking noise and screen flashing!
        const photo = await cameraRef.current.takePictureAsync({
          base64: true,
          quality: 0.1,
          shutterSound: false
        });

        const base64Data = 'data:image/jpeg;base64,' + photo.base64;

        // Display frame immediately on the HTML UI (mimicking real-time WebRTC)
        if (webviewRef.current) {
          webviewRef.current.injectJavaScript(`
              var imgEl = document.getElementById('nativeImageStream');
              if(imgEl) { imgEl.src = '${base64Data}'; }
              true;
            `);
        }

        // POST to backend for AI processing silently over the HTTP LAN
        const response = await fetch(`${SERVER_URL}/process_frame`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: base64Data,
            student_id: data.student_id,
            model_choice: data.model_choice
          })
        });

        const resultData = await response.json();

        // Send alert data back into WebView JavaScript so the HTML UI shakes/flashes!
        if (webviewRef.current) {
          webviewRef.current.injectJavaScript(`
              var imgEl = document.getElementById('nativeImageStream');
              if(imgEl) { imgEl.src = '${base64Data}'; }

              if(window.processBackendAlerts) {
                 window.processBackendAlerts(${JSON.stringify(resultData)});
              }
              if(typeof window.isCapturing !== 'undefined') {
                 window.isCapturing = false; // unlock for next frame
              }
              true;
            `);
        }
        setIsProcessing(false);
      } else if (data.action === 'take_single_picture' && cameraRef.current && !isProcessing) {
        setIsProcessing(true);
        // Take a picture for verify/register endpoints silently
        const photo = await cameraRef.current.takePictureAsync({
          base64: true,
          quality: 0.5,
          shutterSound: false
        });

        const base64Data = 'data:image/jpeg;base64,' + photo.base64;

        if (webviewRef.current) {
          webviewRef.current.injectJavaScript(`
              if(window.receivePicture) {
                 window.receivePicture('${base64Data}');
              }
              true;
            `);
        }
        setIsProcessing(false);
      }
    } catch (err) {
      console.log("Error handling native message:", err);
      setIsProcessing(false);
    }
  };

  if (!permission) {
    return <View style={styles.center}><ActivityIndicator size="large" color="#4F46E5" /></View>;
  }

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>Camera Access Required</Text>
        <Text style={styles.subText} onPress={requestPermission}>Tap to grant permission</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* Hidden Native Camera acts as our isolated capturing engine */}
      <View style={{ width: 1, height: 1, overflow: 'hidden' }}>
        <CameraView style={{ width: 10, height: 10 }} facing="front" ref={cameraRef} animateShutter={false} />
      </View>

      {/* The main interface continues to run the Flask web app flawlessly */}
      <WebView
        ref={webviewRef}
        source={{ uri: SERVER_URL }}
        style={styles.webview}
        injectedJavaScript={injectedScript}
        injectedJavaScriptBeforeContentLoaded={`window.isExpoApp = true; true;`}
        onMessage={onMessage}
        javaScriptEnabled={true}
        domStorageEnabled={true}
        allowsInlineMediaPlayback={true}
        mediaPlaybackRequiresUserAction={false}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  webview: { flex: 1 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  errorText: { fontSize: 18, color: '#ef4444', marginBottom: 10 },
  subText: { fontSize: 16, color: '#4F46E5', textDecorationLine: 'underline', padding: 10 }
});
