import React, { useState, useEffect } from 'react';
import { StyleSheet, View, Text, ActivityIndicator, Alert, SafeAreaView, Platform } from 'react-native';
import { WebView } from 'react-native-webview';
import { Camera } from 'expo-camera';

// NOTE: Replace this IP with the computer's actual local IPv4 address
// (e.g., usually 192.168.x.x or 10.0.x.x)
const SERVER_URL = 'http://192.168.1.100:7860';

export default function App() {
  const [hasPermission, setHasPermission] = useState(null);

  useEffect(() => {
    (async () => {
      // Request native camera and microphone permissions on the mobile device
      const cameraStatus = await Camera.requestCameraPermissionsAsync();
      const microphoneStatus = await Camera.requestMicrophonePermissionsAsync();

      setHasPermission(
        cameraStatus.status === 'granted' && microphoneStatus.status === 'granted'
      );
    })();
  }, []);

  if (hasPermission === null) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#4F46E5" />
        <Text style={styles.text}>Requesting Camera Permissions...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>No access to camera or microphone.</Text>
        <Text style={styles.subText}>The AI Proctoring app requires these permissions to function.</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <WebView
        source={{ uri: SERVER_URL }}
        style={styles.webview}
        javaScriptEnabled={true}
        domStorageEnabled={true}
        allowsInlineMediaPlayback={true}
        // Essential configuration for WebRTC in WebView
        mediaPlaybackRequiresUserAction={false}
        // Handle permissions requested by the web app (WebRTC)
        onPermissionRequest={(event) => {
          Alert.alert(
            "Camera Request",
            "The Proctoring Service needs to access your camera",
            [
              { text: "Deny", onPress: () => event.deny(), style: "cancel" },
              { text: "Grant", onPress: () => event.grant() }
            ]
          );
        }}
        onError={(syntheticEvent) => {
          const { nativeEvent } = syntheticEvent;
          console.error("WebView error: ", nativeEvent);
        }}
        renderError={(errorName) => (
          <View style={styles.center}>
            <Text style={styles.errorText}>Could not connect to the Backend.</Text>
            <Text style={styles.subText}>Make sure your PC's Flask Server is running on 0.0.0.0:7860</Text>
            <Text style={styles.subText}>Check that "{SERVER_URL}" is your PC's correct IP Address.</Text>
          </View>
        )}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  webview: {
    flex: 1,
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#f8fafc',
  },
  text: {
    marginTop: 10,
    fontSize: 16,
    color: '#334155',
  },
  errorText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ef4444',
    marginBottom: 10,
    textAlign: 'center',
  },
  subText: {
    fontSize: 14,
    color: '#64748b',
    textAlign: 'center',
  }
});
