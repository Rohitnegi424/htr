
import React from 'react';

export default function Result({ text }) {
  if (!text) return null;
  return (
    <div className="result-box">
      <h3>Recognized Text</h3>
      <p>{text}</p>
    </div>
  );
}
