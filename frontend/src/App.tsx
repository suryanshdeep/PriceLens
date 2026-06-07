import { Navigate, Route, Routes } from "react-router-dom";

import Navbar from "./components/Navbar";
import About from "./pages/About";
import Analysis from "./pages/Analysis";
import Home from "./pages/Home";
import Predict from "./pages/Predict";
import SimilarSearch from "./pages/SimilarSearch";

export default function App() {
  return (
    <div className="min-h-screen bg-[#f7f8f6]">
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/predict" replace />} />
          <Route path="/home" element={<Home />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/similar-search" element={<SimilarSearch />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
    </div>
  );
}
