import React, { useEffect, useState } from 'react'
import SideBar from '../components/SideBar'
import Card from '../components/Card';
import axios from 'axios';

const Warnings = () => {
  const [warning, setWarning] = useState([]);


  useEffect(() => {
    fetchRecentActivity();
  }, []);
  const fetchRecentActivity = async () => {
    try {
      const response = await axios.get(`${process.env.REACT_APP_BACKEND_URL}api/errorLogs/all/Warning`, { 
        headers: { 'ngrok-skip-browser-warning': 'true' }
      });
      setWarning(response.data); 
    } catch (error) {
      console.error("Error fetching recent activity:", error);
    }
  };

  return (
    <div className='flex min-h-screen'>
      <SideBar/>
      <div className="w-9/12 p-8 overflow-y-auto">
              <div className="grid grid-cols-1 gap-4">
                  {warning.map((activity, index) => (
                    <div key={index}>
                      <Card
                        title={activity.createdAt}
                        status={activity.resolved || "Warning"}
                        errorLog={activity.log}
                        solution="No solution needed for this entry."
                      />
                    </div>
                  ))}
              </div>
            </div>
    </div>
  )
}

export default Warnings