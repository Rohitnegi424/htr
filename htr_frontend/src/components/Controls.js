
import React, { useState } from 'react';

export default function Controls({ text }) {
  const [lang, setLang] = useState('en-US');

  const speak = () => {
    if (!text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    speechSynthesis.speak(utterance);
  };

  return (
    <div className="controls">
      <select onChange={(e) => setLang(e.target.value)}>
        <option value="en-US">English</option>
        <option value="hi-IN">Hindi</option>
      </select>
      <button onClick={speak}>Read Aloud</button>
    </div>
  );
}
