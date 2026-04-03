import { Outlet, NavLink } from "react-router-dom";
import { useControlState } from "@/hooks/useControlState";
import { StateContext } from "@/context";

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-1 text-xs ${
    isActive
      ? "text-orange-400 border border-orange-800 bg-orange-950"
      : "text-gray-500 hover:text-gray-300"
  }`;

export function Layout() {
  const ctrl = useControlState();

  return (
    <StateContext.Provider value={ctrl}>
      <div className="flex flex-col h-screen">
        <nav
          className="flex items-center gap-4 px-4 py-2 border-b"
          style={{
            background: "var(--bg-secondary)",
            borderColor: "var(--border)",
          }}
        >
          <span style={{ color: "var(--accent-orange)", fontWeight: "bold" }}>
            ANT-MNIST
          </span>

          <NavLink to="/" className={navLinkClass}>
            Overview
          </NavLink>
          <NavLink to="/detectors" className={navLinkClass}>
            Detectors
          </NavLink>

          <div className="ml-auto flex items-center gap-4 text-xs">
            <span className="text-gray-500">
              Cycle{" "}
              <span className="text-white">{ctrl.state.cycle}</span>
            </span>
            <span className="text-gray-500">
              Test{" "}
              <span className="text-white">
                {ctrl.state.test_acc
                  ? (ctrl.state.test_acc * 100).toFixed(1) + "%"
                  : "-"}
              </span>
            </span>
            <span className="text-gray-500">
              Best{" "}
              <span style={{ color: "var(--accent-green)" }}>
                {ctrl.state.best_test_acc
                  ? (ctrl.state.best_test_acc * 100).toFixed(1) + "%"
                  : "-"}
              </span>
            </span>
            <span
              className={`w-2 h-2 rounded-full ${
                ctrl.connected ? "bg-green-500" : "bg-red-500"
              }`}
            />
          </div>
        </nav>

        <div className="flex-1 overflow-auto p-2">
          <Outlet />
        </div>
      </div>
    </StateContext.Provider>
  );
}
