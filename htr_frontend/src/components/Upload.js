
import React from 'react';

export default function Upload({ setFile }) {
  return (
    <div className="upload-box">
      <input type="file" accept="image/*"
        onChange={(e) => setFile(e.target.files[0])} />
    </div>
  );
}
