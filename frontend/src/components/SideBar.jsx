import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faChartLine,
  faCheck,
  faCircleExclamation,
  faExclamationTriangle,
} from "@fortawesome/free-solid-svg-icons";
import { Link, useLocation } from "react-router-dom";
import logo from "../utils/logo.jpeg"
import { faUpload } from '@fortawesome/free-solid-svg-icons';
const menuItems = [
  {
    icon: faChartLine,
    color: "#000000",
    text: "Dashboard",
    url: "/",
  },
  {
    icon: faExclamationTriangle,
    color: "#f60404",
    text: "Critical errors",
    url: "/criticalerrors",
  },
  {
    icon: faCheck,
    color: "#04f608",
    text: "Non-critical Error",
    url: "/noncriticalerrors",
  },
  {
    icon: faCircleExclamation,
    color: "#FFD43B",
    text: "Warnings",
    url: "/warnings",
  },
  {
    icon: faUpload,
    color: "#213454",
    text: "Upload",
    url: "/upload",
  },
];

const SideBar = () => {
  const location = useLocation(); // Hook to get the current location

  return (
    <>
      <aside className=" min-w-4/12 bg-white shadow-md mr-1">
        <div className="p-4 flex items-center space-x-2">
          <img className="h-28" src={logo}/>
        </div>
        <nav className="mt-8">
          {menuItems.map((item, index) => (
            <Link key={index} to={item.url}>
              <div
                className={`flex items-center px-4 py-2 text-gray-700 hover:bg-gray-200 
                ${location.pathname === item.url ? "bg-blue-100 text-blue-500" : ""}`}
              >
                <FontAwesomeIcon
                  icon={item.icon}
                  style={{ color: item.color }}
                  className="mx-2"
                />
                {item.text}
              </div>
            </Link>
          ))}
        </nav>
      </aside>
    </>
  );
};

export default SideBar;
