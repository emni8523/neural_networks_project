import {MediaRecorder, register} from 'extendable-media-recorder';
import {connect} from 'extendable-media-recorder-wav-encoder';
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let currentStream = null;

await register(await connect());

export async function toggleRecording(micButton) {
  micButton.classList.toggle('active');
  
  if (!isRecording) {
    document.querySelector('.div').textContent = "Listening...";
    document.querySelector('.rectangle-2 p').textContent = "";
      try {
      currentStream = await navigator.mediaDevices.getUserMedia({ audio: true }); // Store stream
      mediaRecorder = new MediaRecorder(currentStream, {
        mimeType: 'audio/wav'
      });
      
      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };
      
      mediaRecorder.onstop = async () => {

        const mimeType = mediaRecorder.mimeType;
        const audioBlob = new Blob(audioChunks, { type: mimeType });
        audioChunks = [];
        
        
        const audioUrl = URL.createObjectURL(audioBlob);
        const a = document.createElement('a');
        a.href = audioUrl;
        a.download = 'recording.wav';
        

        await sendAudioToServer(audioBlob);
      };
      
      mediaRecorder.start();
      isRecording = true;
      console.log('Recording started');
    } catch (err) {
      console.error('Error accessing microphone:', err);
      micButton.classList.remove('active');
    }
  } else {
    mediaRecorder.stop();
    currentStream.getTracks().forEach(track => track.stop());
    isRecording = false;
    console.log('Recording stopped');
  }
}

async function sendAudioToServer(audioBlob) {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.wav');
  
  try {
      const response = await fetch('http://localhost:8000/process-audio', {
        method: 'POST',
        body: formData
      });
    
    const result = await response.json();
    console.log('Server response:', result);
    
    updateUI(result);
  } catch (err) {
    console.error('Error sending audio:', err);
    document.querySelector('.div').textContent = "Error: " + err.message;
  }
}

// To be used later for parsing
function updateUI(result) {
  const originalText = document.querySelector('.div');
  const filteredText = document.querySelector('.rectangle-2 p');
  
  if (result.original_text) {
    originalText.textContent = result.original_text;
  }
  if (result.filtered_text) {
    filteredText.textContent = result.filtered_text;
  }
}

export { isRecording };