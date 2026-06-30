/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"]
      },
      colors: {
        ink: "#17202a",
        cloud: "#f5f7fb",
        panel: "#ffffff",
        line: "#d8dee9",
        teal: "#0f9f9a",
        cobalt: "#2d6cdf",
        amber: "#d99025",
        rose: "#d1495b"
      },
      boxShadow: {
        soft: "0 10px 30px rgba(23, 32, 42, 0.08)"
      }
    }
  },
  plugins: []
};
