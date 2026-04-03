import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { OverviewPage } from "@/pages/OverviewPage";
import { DetectorsPage } from "@/pages/DetectorsPage";

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<OverviewPage />} />
          <Route path="/detectors" element={<DetectorsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
