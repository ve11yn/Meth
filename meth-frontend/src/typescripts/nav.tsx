import { Link, useLocation } from "react-router-dom";
import "../styling/index.css";

const Navigation: React.FC = () => {
  const { pathname } = useLocation();

  return (
    <nav className="relative w-full">
      <div className="fixed top-6 left-1/2 transform -translate-x-1/2 z-50">
        <div className="flex items-center justify-between bg-white/80 backdrop-blur-xl border border-gray-200/50 rounded-full px-8 py-4 shadow-2xl shadow-black/5">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-3 group">
            <div className="relative">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-slate-900 via-slate-700 to-slate-900 flex items-center justify-center relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent"></div>
                <span className="text-white text-sm font-light tracking-wider relative z-10">M</span>
              </div>
              <div className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-amber-400 shadow-lg shadow-amber-400/50"></div>
            </div>
            <div className="flex flex-col">
              <span className="text-lg font-light tracking-wide text-slate-900 group-hover:text-slate-700 transition-colors duration-300">Meth</span>
              <div className="w-0 h-px bg-gradient-to-r from-slate-900 to-slate-400 group-hover:w-full transition-all duration-500"></div>
            </div>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center space-x-12 ml-16">
            <Link
              to="/"
              className={`relative text-sm font-light tracking-wide transition-all duration-300 group ${
                pathname === "/" ? "text-slate-900" : "text-slate-500 hover:text-slate-900"
              }`}
            >
              <span className="relative z-10">Home</span>
              <div className="absolute inset-0 -z-10">
                <div
                  className={`absolute bottom-0 left-0 h-px bg-gradient-to-r from-slate-900 to-transparent transition-all duration-300 ${
                    pathname === "/" ? "w-full" : "w-0 group-hover:w-full"
                  }`}
                ></div>
                <div className="absolute -bottom-2 left-0 w-1 h-1 rounded-full bg-slate-900 opacity-0 group-hover:opacity-100 transition-all duration-300 transform translate-y-1 group-hover:translate-y-0"></div>
              </div>
            </Link>

            <Link
              to="/about"
              className={`relative text-sm font-light tracking-wide transition-all duration-300 group ${
                pathname === "/about" ? "text-slate-900" : "text-slate-500 hover:text-slate-900"
              }`}
            >
              <span className="relative z-10">About</span>
              <div className="absolute inset-0 -z-10">
                <div
                  className={`absolute bottom-0 left-0 h-px bg-gradient-to-r from-slate-900 to-transparent transition-all duration-300 ${
                    pathname === "/about" ? "w-full" : "w-0 group-hover:w-full"
                  }`}
                ></div>
                <div className="absolute -bottom-2 left-0 w-1 h-1 rounded-full bg-slate-900 opacity-0 group-hover:opacity-100 transition-all duration-300 transform translate-y-1 group-hover:translate-y-0"></div>
              </div>
            </Link>
          </div>
        </div>

        {/* Subtle floating shadow */}
        <div className="absolute inset-0 bg-gradient-to-r from-slate-900/5 via-slate-500/5 to-slate-900/5 rounded-full blur-xl transform translate-y-2 -z-10"></div>
      </div>

      {/* Background accent */}
      <div className="fixed top-0 right-0 w-96 h-96 bg-gradient-to-bl from-amber-50 to-transparent rounded-full blur-3xl opacity-30 -z-20"></div>
      <div className="fixed top-20 left-0 w-64 h-64 bg-gradient-to-br from-slate-50 to-transparent rounded-full blur-2xl opacity-40 -z-20"></div>
    </nav>
  );
};

export default Navigation;
