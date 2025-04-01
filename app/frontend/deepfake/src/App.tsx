import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./components/Home";
import DeepfakeDetector from "./components/DeepfakeDetector";

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/detector" element={<DeepfakeDetector />} />
      </Routes>
    </Router>
  );
}

export default App;
