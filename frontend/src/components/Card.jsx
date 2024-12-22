import React, { useState } from "react";
import axios from "axios";
import ReactHtmlParser from "html-react-parser";
import Groq from "groq-sdk/index.mjs";
import { marked } from "marked";

const groq = new Groq({
  apiKey: process.env.REACT_APP_GROQ_API_KEY,
  dangerouslyAllowBrowser: true,
});

let parser = new DOMParser();

export async function main() {
  const chatCompletion = await getGroqChatCompletion();
  // Print the completion returned by the LLM.
  console.log(chatCompletion.choices[0]?.message?.content || "");
}

export async function getGroqChatCompletion(content) {
  return groq.chat.completions.create({
    messages: [
      {
        role: "user",
        content,
      },
    ],
    model: "llama3-8b-8192",
  });
}

const Card = ({ title, status, errorLog }) => {
  const [showSolution, setShowSolution] = useState(false);
  const [loading, setLoading] = useState(false); // Loading state
  const [solution, setSolution] = useState(""); // Solution state

  // Define the question based on the error log
  const question = `${errorLog}, explain the error log from apache server and explain the steps to troubleshoot it?`;
  console.log(status);
  // Handle the click event and make the POST request
  const handleClick = async () => {
    setShowSolution(!showSolution);

    if (!showSolution && !solution) {
      // Only fetch if solution is not already available
      setLoading(true); // Start loader
      try {
        // const response = await axios.post(
        //   `${process.env.REACT_APP_FLASK_URL}ask`,
        //   {
        //     question: question, // Question is passed in the body
        //   },
        //   {
        //     headers: {
        //       'ngrok-skip-browser-warning': 'true', // Custom headers
        //     },
        //   }
        // );

        const chatCompletion = await getGroqChatCompletion(question);
        const response = chatCompletion.choices[0]?.message?.content || null;

        // Update the solution with the response
        setSolution(marked(response) || "No solution found.");
      } catch (error) {
        console.error("Error during POST request:", error);
        setSolution("Failed to fetch solution.");
      } finally {
        setLoading(false); // Stop loader
      }
    }
  };
  console.log(solution);
  // Determine tag color based on status
  const statusColors = {
    Critical: "bg-red-500 text-white",
    Noncritical: "bg-green-500 text-white",
    Warning: "bg-yellow-500 text-black",
  };

  return (
    <div className="w-full flex flex-col border rounded-lg shadow-md p-6 mb-4">
      {/* Card Header */}
      <div className="flex justify-between items-center mb-4">
        {/* Title */}
        <h2 className="text-xl font-bold">{title}</h2>
        {/* Status Tag */}
        <span
          className={`text-sm font-semibold py-1 px-3 rounded-full ${
            statusColors[status] || "bg-gray-500 text-white"
          }`}
        >
          {status}
        </span>
      </div>

      {/* Error Log */}
      <p className="text-gray-600 mb-4">{errorLog}</p>

      {/* Enter Button */}
      <button
        className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 w-28"
        onClick={handleClick}
      >
        {showSolution ? "Hide Solution" : "Enter"}
      </button>

      {/* Solution Section (conditionally rendered) */}
      {showSolution && (
        <div className="mt-4 p-4 bg-gray-100 rounded-lg border">
          <h3 className="text-lg font-semibold text-green-600 mb-2">
            Solution:
          </h3>

          {/* Show Loader while fetching */}
          {loading ? (
            <p className="text-gray-700">Loading solution...</p>
          ) : (
            // Split the solution by new lines and render each line
            <div>{ReactHtmlParser(solution)}</div>
          )}
        </div>
      )}
    </div>
  );
};

export default Card;
