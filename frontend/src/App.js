import './App.css'
import Dashboard from './components/Dashboard';
import SignIn from './components/SignIn'
import SignUp from './components/SignUp'
import { Route, Routes } from "react-router-dom";
import NonCriticalErrors from './routes/NonCriticalErrors';
import CriticalErrors from './routes/CriticalErrors';
import Warnings from './routes/Warnings';
import SideBar from './components/SideBar';
import FileUpload from "./components/FileUpload"
function App() {

  return (
    <div>
        <div>
        <Routes>


        <Route path="/" element={<Dashboard/>}></Route>
        <Route path="/criticalerrors" element={<CriticalErrors/>}></Route>
        <Route path="/noncriticalerrors" element={<NonCriticalErrors/>}></Route>
        <Route path="/warnings" element={<Warnings/>}></Route>
        <Route path="/upload" element={<FileUpload/>}></Route>
      </Routes> 
        </div>
       
    </div>
  )
}

export default App
