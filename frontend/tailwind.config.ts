import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0f172a",
        panel: "#ffffff",
        line: "#d8e0ec",
        accent: "#0f8a5f",
        pitch: "#0f8a5f"
      }
    }
  },
  plugins: []
};

export default config;
