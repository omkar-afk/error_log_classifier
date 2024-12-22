import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import SideBar from './SideBar';

const Upload = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const navigate = useNavigate(); 
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("Please upload a CSV file");
      return;
    }
    console.log(file)
    const formData = new FormData();
    formData.append('file', file);  
    if(file){
      navigate("/")
    }
    setUploading(true);
    try {
      const response = await axios.post(`${process.env.REACT_APP_BACKEND_URL}api/multer/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const response2 = await axios.post(`${process.env.REACT_APP_BACKEND_URL}api/errorLogs/add`,{path:response.data.filePath,id:"6713ae2fd0b070ed8e0f144f"})
      console.log(response2)
      if(response2.status === 207){
        response2.data.results.total = response2.data.results.processed + response2.data.results.failed
        response2.data.results.successful = response2.data.results.processed
      }
      alert(`Total Logs Proccessed ${response2.data.results.total} , Logs Successfull proccessed ${response2.data.results.successful} and log proccessing failed ${response2.data.results.failed}`)
      setMessage('File uploaded successfully!');
      console.log("Response:", response.data);
    } catch (error) {
      setMessage('Error uploading file');
      console.error("Error:", error.response ? error.response.data : error.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className='flex'>
    <SideBar/>
    <div className="min-h-screen flex items-center justify-center  ml-60">
        
      <div className="w-full max-w-md bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold text-center mb-4">Upload Error-Log text file</h2>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <input
              type="file"
              accept="text/plain"
              onChange={handleFileChange}
              className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring focus:ring-blue-200"
              required
            />
          </div>
          <button
            type="submit"
            className={`w-full bg-blue-500 text-white py-2 rounded-md hover:bg-blue-600 focus:outline-none ${uploading ? 'opacity-50 cursor-not-allowed' : ''}`}
            disabled={uploading}
          >
            {uploading ? 'Uploading...' : 'Upload'}
          </button>
        </form>
        {message && <p className="mt-4 text-center text-red-500">{message}</p>}
      </div>
    </div>
    </div>
  );
};

export default Upload;
