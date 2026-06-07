/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        ink: "#18202b",
        steel: "#3f5f73",
        mint: "#0f9f7a",
        amber: "#c9821a",
        coral: "#d95f59",
      },
    },
  },
  plugins: [],
};
