
import React, { useState } from 'react';
import Upload from './components/Upload';
import Result from './components/Result';
import Controls from './components/Controls';
import { sendImage } from './services/api';

export default function App() {
  const [file, setFile] = useState(null);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const res = await sendImage(file);
      setText(res.text || 'No text detected');
    } catch (error) {
      console.error('Prediction failed', error);
      setText('Prediction failed. Check backend server and try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    setFile(null);
    setText("");
  };

  return (
    <div className="container">
      <h1>HTR AI App</h1>

      <Upload setFile={setFile} />

      <div className="btn-group">
        <button onClick={handlePredict}>Recognize</button>
        <button onClick={handleCancel} className="cancel">Cancel</button>
      </div>

      {loading && <p>Processing...</p>}

      <Result text={text} />

      <Controls text={text} />
    </div>
  );
}
