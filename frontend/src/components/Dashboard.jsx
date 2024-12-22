import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  faCheck,
  faCircleExclamation,
  faExclamationTriangle,
} from "@fortawesome/free-solid-svg-icons";
import Card from "./Card";
import SideBar from "./SideBar";

const Dashboard = () => {
  const [stats, setStats] = useState([
    {
      icon: faExclamationTriangle,
      label: "Critical Errors",
      value: "0", // default value, will be updated by API
      color: "#f60404",
      bgcolor: "bg-[#C1E0FF]",
    },
    {
      icon: faCheck,
      label: "Non-critical Errors",
      value: "0", // default value, will be updated by API
      color: "#04f608",
      bgcolor: "bg-[#FEF3D7]",
    },
    {
      icon: faCircleExclamation,
      label: "Warnings",
      value: "0", // default value, will be updated by API
      color: "#FFD43B",
      bgcolor: "bg-[#EFE1FC]",
    },
  ]);

  const [recentActivity, setRecentActivity] = useState([]);

  useEffect(() => {
    fetchRecentActivity();
    fetchErrorStats(); // Fetch the stats when the component loads
  }, []);

  const fetchRecentActivity = async () => {
    try {
      const response = await axios.get(`${process.env.REACT_APP_BACKEND_URL}api/errorLogs/top10`, { 
        headers: { 'ngrok-skip-browser-warning': 'true' }
      });

      // Check if response data is an array
      // console.log(response.data.top10); // Inspect the API response
      console.log(response.data.top10)
        setRecentActivity(response.data.top10); 
    } catch (error) {
      console.error("Error fetching recent activity:", error);
    }
  };

  const fetchErrorStats = async () => {
    try {
      const response = await axios.get(`${process.env.REACT_APP_BACKEND_URL}api/errorLogs/count`, { 
        headers: { 'ngrok-skip-browser-warning': 'true' }
      });

      // console.log(response.data.count); // Inspect the API response

      const { critical, nonCritical, warning } = response.data.count;

      setStats([
        {
          icon: faExclamationTriangle,
          label: "Critical Errors",
          value: critical,
          color: "#f60404",
          bgcolor: "bg-[#C1E0FF]",
        },
        {
          icon: faCheck,
          label: "Non-critical Errors",
          value: nonCritical,
          color: "#04f608",
          bgcolor: "bg-[#FEF3D7]",
        },
        {
          icon: faCircleExclamation,
          label: "Warnings",
          value: warning,
          color: "#FFD43B",
          bgcolor: "bg-[#EFE1FC]",
        },
      ]);
    } catch (error) {
      console.error("Error fetching error stats:", error);
    }
  };

  return (
    <div className="flex min-h-screen">
      <SideBar />
      <div className="w-9/12 bg-white">
        <main className="flex-1 p-8">
          <div className="flex justify-between items-center mt-6">
            <h1 className="text-4xl font-semibold">Dashboard</h1>
          </div>

          <div className="grid grid-cols-3 gap-10 mb-8 mt-14">
            {stats.map((stat, index) => (
              <div
                key={index}
                className={`p-4 rounded-lg ${stat.bgcolor} flex flex-col items-center justify-center`}
              >
                <FontAwesomeIcon
                  icon={stat.icon}
                  style={{ color: stat.color }}
                  className="mb-2"
                  size="2x"
                />
                <h2 className="text-2xl font-semibold">{stat.value}</h2>
                <p className="text-gray-600">{stat.label}</p>
              </div>
            ))}
          </div>

          <div className="mt-5">
            <h2 className="text-xl font-semibold p-4">Recent Activity</h2>
            <div className="overflow-y-auto">
              <div className="grid grid-cols-1 gap-4">
                {/* Ensure recentActivity is an array before mapping */}
                {Array.isArray(recentActivity) && recentActivity.length > 0 ? (
                  recentActivity.map((activity, index) => (
                    <div key={index}>
                      <Card
                        title={activity.createdAt}
                        status={activity.type}
                        errorLog={activity.log}
                        solution="No solution needed for this entry."
                      />
                    </div>
                  ))
                ) : (
                  <p>No recent activity available</p>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default Dashboard;
