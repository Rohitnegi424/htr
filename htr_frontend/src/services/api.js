
import axios from 'axios';

export const sendImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const res = await axios.post('/predict', formData);
  return res.data;
};
