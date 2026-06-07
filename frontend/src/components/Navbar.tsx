import { BarChart3, Boxes, Info, Search, Sparkles } from "lucide-react";
import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/predict", label: "Predict", icon: Sparkles },
  { to: "/analysis", label: "Analysis", icon: BarChart3 },
  { to: "/similar-search", label: "Similar", icon: Search },
  { to: "/home", label: "Overview", icon: Boxes },
  { to: "/about", label: "About", icon: Info },
];

export default function Navbar() {
  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
        <div>
          <p className="text-xl font-semibold tracking-normal text-ink">PriceLens</p>
          <p className="text-sm text-slate-500">Product price intelligence workspace</p>
        </div>
        <nav className="flex flex-wrap gap-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition ${
                  isActive
                    ? "bg-ink text-white"
                    : "text-slate-600 hover:bg-slate-100 hover:text-ink"
                }`
              }
            >
              <item.icon className="h-4 w-4" aria-hidden="true" />
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
