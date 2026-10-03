import { jsx as Xs } from "react/jsx-runtime";
import { forwardRef as qs, useRef as Xt, useImperativeHandle as Rs, useId as Ws, useLayoutEffect as _s, useEffect as Xe } from "react";
const le = {
  clover: { label: "Clover", color: "#35B8FF", face: "eyes", faceX: 50, faceY: 50, faceScale: 1 },
  flower: { label: "Flower", color: "#2FCB7A", face: "eyes", faceX: 50, faceY: 51, faceScale: 0.95 },
  triangle: { label: "Triangle", color: "#DC48FF", face: "eyes", faceX: 50, faceY: 61, faceScale: 0.9 },
  square: { label: "Square", color: "#35B8FF", face: "eyes", faceX: 50, faceY: 50, faceScale: 1 },
  blob: { label: "Blob", color: "#2FCB7A", face: "eyes", faceX: 49.5, faceY: 50, faceScale: 1 },
  ghost: { label: "Ghost", color: "#F4F2FA", face: "eyes", faceX: 50, faceY: 48, faceScale: 0.95 },
  circle: { label: "Circle", color: "#9A62FF", face: "eyes", faceX: 50, faceY: 50, faceScale: 1 },
  drop: { label: "Drop", color: "#1ED3C6", face: "eyes", faceX: 50, faceY: 62, faceScale: 0.9 },
  star: { label: "Star", color: "#FFD32B", face: "eyes", faceX: 50, faceY: 52, faceScale: 0.82 },
  droid: { label: "Droid", color: "#D5DBEA", face: "eyes", faceX: 50, faceY: 60, faceScale: 0.95 },
  mech: { label: "Mech", color: "#95A6C4", face: "eyes", faceX: 50, faceY: 59, faceScale: 1 },
  alien: { label: "Alien", color: "#9BE85A", face: "eyes", faceX: 50, faceY: 45, faceScale: 1.05 },
  hexagon: { label: "Hexagon", color: "#FF2A2A", face: "eyes", faceX: 50, faceY: 50, faceScale: 0.95 },
  cat: { label: "Cat", color: "#FF8C42", face: "eyes", faceX: 50, faceY: 58, faceScale: 1 },
  cloud: { label: "Cloud", color: "#CFE6FF", face: "eyes", faceX: 50, faceY: 58, faceScale: 0.95 },
  pill: { label: "Pill", color: "#7B77F0", face: "eyes", faceX: 50, faceY: 50, faceScale: 0.9 },
  pebble: { label: "Pebble", color: "#2FCB7A", face: "eyes", faceX: 50, faceY: 50, faceScale: 0.95 },
  puddle: { label: "Puddle", color: "#FF2A2A", face: "eyes", faceX: 50, faceY: 50, faceScale: 0.95 }
}, Ds = Object.keys(le), Wa = ["eyes", "mouth"], _a = ["default", "working", "sleeping"], Da = Object.fromEntries(
  Ds.map((t) => [t, le[t].color])
), qe = {
  default: "idle",
  working: "working",
  sleeping: "sleeping"
}, Re = {
  clover: "M26.53 22.38A25 25 0 0 1 73.47 22.38A7 7 0 0 0 77.62 26.53A25 25 0 0 1 77.62 73.47A7 7 0 0 0 73.47 77.62A25 25 0 0 1 26.53 77.62A7 7 0 0 0 22.38 73.47A25 25 0 0 1 22.38 26.53A7 7 0 0 0 26.53 22.38Z",
  flower: "M31.2 19.82A20.5 20.5 0 0 1 68.8 19.82A5 5 0 0 0 72.9 22.8A20.5 20.5 0 0 1 84.51 58.55A5 5 0 0 0 82.95 63.37A20.5 20.5 0 0 1 52.54 85.47A5 5 0 0 0 47.46 85.47A20.5 20.5 0 0 1 17.05 63.37A5 5 0 0 0 15.49 58.55A20.5 20.5 0 0 1 27.1 22.8A5 5 0 0 0 31.2 19.82Z",
  triangle: "M38.75 27.43A13 13 0 0 1 61.25 27.43L82.7 64.49A13 13 0 0 1 71.45 84L28.55 84A13 13 0 0 1 17.3 64.49L38.75 27.43Z",
  square: "M93 50C93 53.66 92.96 58.23 92.88 60.97C92.8 63.71 92.67 64.81 92.51 66.44C92.35 68.08 92.15 69.44 91.9 70.77C91.66 72.1 91.37 73.3 91.04 74.44C90.72 75.58 90.35 76.63 89.94 77.63C89.53 78.63 89.07 79.55 88.58 80.43C88.08 81.31 87.54 82.13 86.96 82.9C86.37 83.67 85.75 84.39 85.07 85.07C84.39 85.75 83.67 86.37 82.9 86.96C82.13 87.54 81.31 88.08 80.43 88.58C79.55 89.07 78.63 89.53 77.63 89.94C76.63 90.35 75.58 90.72 74.44 91.04C73.3 91.37 72.1 91.66 70.77 91.9C69.44 92.15 68.08 92.35 66.44 92.51C64.81 92.67 63.71 92.8 60.97 92.88C58.23 92.96 53.66 93 50 93C46.34 93 41.77 92.96 39.03 92.88C36.29 92.8 35.19 92.67 33.56 92.51C31.92 92.35 30.56 92.15 29.23 91.9C27.9 91.66 26.7 91.37 25.56 91.04C24.42 90.72 23.37 90.35 22.37 89.94C21.37 89.53 20.45 89.07 19.57 88.58C18.69 88.08 17.87 87.54 17.1 86.96C16.33 86.37 15.61 85.75 14.93 85.07C14.25 84.39 13.63 83.67 13.04 82.9C12.46 82.13 11.92 81.31 11.42 80.43C10.93 79.55 10.47 78.63 10.06 77.63C9.65 76.63 9.28 75.58 8.96 74.44C8.63 73.3 8.34 72.1 8.1 70.77C7.85 69.44 7.65 68.08 7.49 66.44C7.33 64.81 7.2 63.71 7.12 60.97C7.04 58.23 7 53.66 7 50C7 46.34 7.04 41.77 7.12 39.03C7.2 36.29 7.33 35.19 7.49 33.56C7.65 31.92 7.85 30.56 8.1 29.23C8.34 27.9 8.63 26.7 8.96 25.56C9.28 24.42 9.65 23.37 10.06 22.37C10.47 21.37 10.93 20.45 11.42 19.57C11.92 18.69 12.46 17.87 13.04 17.1C13.63 16.33 14.25 15.61 14.93 14.93C15.61 14.25 16.33 13.63 17.1 13.04C17.87 12.46 18.69 11.92 19.57 11.42C20.45 10.93 21.37 10.47 22.37 10.06C23.37 9.65 24.42 9.28 25.56 8.96C26.7 8.63 27.9 8.34 29.23 8.1C30.56 7.85 31.92 7.65 33.56 7.49C35.19 7.33 36.29 7.2 39.03 7.12C41.77 7.04 46.34 7 50 7C53.66 7 58.23 7.04 60.97 7.12C63.71 7.2 64.81 7.33 66.44 7.49C68.08 7.65 69.44 7.85 70.77 8.1C72.1 8.34 73.3 8.63 74.44 8.96C75.58 9.28 76.63 9.65 77.63 10.06C78.63 10.47 79.55 10.93 80.43 11.42C81.31 11.92 82.13 12.46 82.9 13.04C83.67 13.63 84.39 14.25 85.07 14.93C85.75 15.61 86.37 16.33 86.96 17.1C87.54 17.87 88.08 18.69 88.58 19.57C89.07 20.45 89.53 21.37 89.94 22.37C90.35 23.37 90.72 24.42 91.04 25.56C91.37 26.7 91.66 27.9 91.9 29.23C92.15 30.56 92.35 31.92 92.51 33.56C92.67 35.19 92.8 36.29 92.88 39.03C92.96 41.77 93 46.34 93 50Z",
  blob: "M93.92 50C94.18 51.9 94.24 53.9 93.99 55.79C93.73 57.68 93.18 59.61 92.39 61.36C91.6 63.1 90.48 64.78 89.27 66.27C88.07 67.76 86.58 69.08 85.14 70.29C83.7 71.49 82.1 72.5 80.62 73.5C79.14 74.49 77.64 75.34 76.24 76.24C74.84 77.15 73.51 78 72.2 78.93C70.88 79.86 69.66 80.83 68.36 81.8C67.06 82.76 65.78 83.82 64.39 84.74C63 85.67 61.55 86.63 60.01 87.35C58.47 88.08 56.82 88.73 55.15 89.1C53.48 89.47 51.71 89.64 50 89.57C48.29 89.5 46.55 89.16 44.91 88.68C43.27 88.21 41.67 87.47 40.16 86.72C38.65 85.96 37.24 85.02 35.86 84.13C34.48 83.24 33.19 82.29 31.89 81.37C30.58 80.46 29.32 79.56 28.03 78.63C26.74 77.71 25.44 76.82 24.16 75.84C22.88 74.86 21.57 73.87 20.36 72.74C19.15 71.62 17.95 70.42 16.91 69.1C15.88 67.79 14.92 66.35 14.15 64.85C13.38 63.35 12.76 61.73 12.29 60.1C11.81 58.48 11.52 56.78 11.29 55.1C11.06 53.41 10.98 51.71 10.91 50C10.83 48.29 10.83 46.59 10.83 44.84C10.84 43.1 10.85 41.34 10.93 39.53C11.02 37.72 11.09 35.86 11.34 33.99C11.59 32.12 11.88 30.16 12.43 28.31C12.99 26.46 13.69 24.56 14.66 22.88C15.64 21.21 16.86 19.58 18.27 18.27C19.69 16.96 21.4 15.84 23.17 15.03C24.93 14.22 26.95 13.71 28.89 13.43C30.83 13.15 32.9 13.21 34.82 13.36C36.74 13.5 38.66 13.92 40.43 14.28C42.2 14.65 43.87 15.16 45.46 15.54C47.06 15.93 48.52 16.31 50 16.58C51.48 16.85 52.87 17.02 54.32 17.17C55.77 17.32 57.22 17.36 58.71 17.49C60.2 17.62 61.74 17.7 63.28 17.95C64.81 18.19 66.39 18.5 67.91 18.98C69.43 19.46 70.95 20.08 72.38 20.83C73.81 21.58 75.19 22.5 76.5 23.5C77.81 24.5 79.03 25.63 80.22 26.81C81.41 27.99 82.54 29.25 83.65 30.57C84.76 31.89 85.85 33.26 86.89 34.72C87.93 36.18 88.98 37.69 89.9 39.31C90.82 40.92 91.73 42.64 92.4 44.42C93.07 46.2 93.66 48.1 93.92 50Z",
  ghost: "M17 50C17 31.78 31.78 17 50 17C68.22 17 83 31.78 83 50V81.5Q72 93.5 61 81.5Q50 93.5 39 81.5Q28 93.5 17 81.5Z",
  circle: "M50 8A42 42 0 1 1 50 92A42 42 0 1 1 50 8Z",
  drop: "M50 8.5C52.2 8.5 53.4 10.3 56.4 15.2C62.9 25.5 84 44.6 84 61.5C84 80.3 68.8 92 50 92C31.2 92 16 80.3 16 61.5C16 44.6 37.1 25.5 43.6 15.2C46.6 10.3 47.8 8.5 50 8.5Z",
  star: "M45.77 9.7A5 5 0 0 1 54.23 9.7L65.02 26.81A4 4 0 0 0 67.42 28.55L87.02 33.53A5 5 0 0 1 89.63 41.57L76.7 57.12A4 4 0 0 0 75.78 59.94L77.11 80.12A5 5 0 0 1 70.26 85.09L51.48 77.59A4 4 0 0 0 48.52 77.59L29.74 85.09A5 5 0 0 1 22.89 80.12L24.22 59.94A4 4 0 0 0 23.3 57.12L10.37 41.57A5 5 0 0 1 12.98 33.53L32.58 28.55A4 4 0 0 0 34.98 26.81L45.77 9.7Z",
  droid: "M16 50C16 38.95 24.95 30 36 30H64C75.05 30 84 38.95 84 50V70C84 81.05 75.05 90 64 90H36C24.95 90 16 81.05 16 70ZM4 62A7 7 0 1 1 18 62A7 7 0 1 1 4 62ZM82 62A7 7 0 1 1 96 62A7 7 0 1 1 82 62Z",
  mech: "M10 48C10 38.06 18.06 30 28 30H72C81.94 30 90 38.06 90 48V70C90 79.94 81.94 88 72 88H28C18.06 88 10 79.94 10 70ZM3 54C3 51.79 4.79 50 7 50H11V72H7C4.79 72 3 70.21 3 68ZM89 50H93C95.21 50 97 51.79 97 54V68C97 70.21 95.21 72 93 72H89Z",
  alien: "M50 10C70 10 83 27 83 46C83 65 64 92 50 92C36 92 17 65 17 46C17 27 30 10 50 10Z",
  hexagon: "M91.4 45.5A9 9 0 0 1 91.4 54.5L74.6 83.61A9 9 0 0 1 66.8 88.11L33.2 88.11A9 9 0 0 1 25.4 83.61L8.6 54.5A9 9 0 0 1 8.6 45.5L25.4 16.39A9 9 0 0 1 33.2 11.89L66.8 11.89A9 9 0 0 1 74.6 16.39L91.4 45.5Z",
  cat: "M50 20A36 36 0 1 1 50 92A36 36 0 1 1 50 20ZM24.72 44.64A4.5 4.5 0 0 1 17.35 40.67L20.21 16.66A4.5 4.5 0 0 1 26.86 13.26L42.3 21.83A4.5 4.5 0 0 1 43.02 29.21L24.72 44.64ZM82.65 40.67A4.5 4.5 0 0 1 75.28 44.64L56.98 29.21A4.5 4.5 0 0 1 57.7 21.83L73.14 13.26A4.5 4.5 0 0 1 79.79 16.66L82.65 40.67Z",
  cloud: "M19 44A25 25 0 1 1 69 44A25 25 0 1 1 19 44ZM47 50A21 21 0 1 1 89 50A21 21 0 1 1 47 50ZM7 68A17 17 0 1 1 41 68A17 17 0 1 1 7 68ZM32 73A18 18 0 1 1 68 73A18 18 0 1 1 32 73ZM60 70A16 16 0 1 1 92 70A16 16 0 1 1 60 70Z",
  pill: "M28 28H72C84.15 28 94 37.85 94 50C94 62.15 84.15 72 72 72H28C15.85 72 6 62.15 6 50C6 37.85 15.85 28 28 28Z",
  pebble: "M94.6 50C94.71 51.92 94.56 53.93 94.17 55.82C93.79 57.7 93.13 59.6 92.28 61.33C91.44 63.06 90.32 64.72 89.1 66.19C87.87 67.67 86.41 69.01 84.92 70.16C83.42 71.31 81.75 72.29 80.11 73.1C78.47 73.92 76.73 74.55 75.06 75.06C73.39 75.58 71.7 75.91 70.1 76.2C68.5 76.48 66.93 76.62 65.45 76.76C63.96 76.9 62.55 76.94 61.2 77.03C59.84 77.11 58.57 77.16 57.31 77.26C56.05 77.36 54.86 77.48 53.64 77.63C52.42 77.79 51.24 77.98 50 78.18C48.76 78.39 47.52 78.64 46.2 78.85C44.89 79.06 43.53 79.3 42.11 79.46C40.69 79.61 39.2 79.75 37.67 79.77C36.14 79.78 34.54 79.74 32.95 79.54C31.35 79.33 29.7 79.03 28.1 78.55C26.49 78.07 24.86 77.45 23.33 76.67C21.8 75.88 20.29 74.94 18.92 73.85C17.54 72.77 16.23 71.51 15.08 70.16C13.94 68.8 12.9 67.29 12.04 65.72C11.17 64.16 10.45 62.46 9.91 60.74C9.36 59.03 8.98 57.22 8.76 55.43C8.55 53.64 8.5 51.8 8.6 50C8.71 48.2 8.98 46.39 9.38 44.65C9.79 42.91 10.35 41.19 11.02 39.56C11.7 37.92 12.52 36.34 13.43 34.85C14.34 33.37 15.39 31.96 16.5 30.66C17.61 29.36 18.84 28.15 20.11 27.07C21.39 25.98 22.76 25 24.15 24.15C25.54 23.3 27.01 22.56 28.48 21.95C29.94 21.33 31.46 20.85 32.95 20.46C34.44 20.08 35.95 19.83 37.43 19.66C38.91 19.48 40.38 19.43 41.81 19.43C43.24 19.43 44.64 19.53 46.01 19.66C47.37 19.78 48.7 19.98 50 20.18C51.3 20.39 52.57 20.63 53.83 20.87C55.1 21.11 56.34 21.37 57.6 21.62C58.87 21.87 60.13 22.12 61.43 22.39C62.74 22.66 64.07 22.93 65.45 23.24C66.83 23.56 68.25 23.88 69.72 24.3C71.19 24.72 72.71 25.17 74.25 25.75C75.78 26.34 77.37 27 78.91 27.81C80.46 28.63 82.04 29.56 83.5 30.66C84.97 31.75 86.43 33 87.7 34.38C88.98 35.77 90.19 37.32 91.17 38.97C92.14 40.62 92.98 42.43 93.56 44.27C94.13 46.1 94.5 48.08 94.6 50Z",
  puddle: "M85.34 50C85.65 51.54 85.9 53.13 86.01 54.74C86.11 56.35 86.11 58 85.95 59.63C85.8 61.26 85.51 62.93 85.07 64.52C84.62 66.12 84.03 67.72 83.3 69.23C82.58 70.74 81.7 72.21 80.72 73.57C79.74 74.94 78.62 76.23 77.43 77.43C76.24 78.62 74.93 79.72 73.58 80.73C72.23 81.74 70.79 82.65 69.33 83.48C67.86 84.31 66.34 85.04 64.79 85.71C63.24 86.37 61.65 86.95 60.04 87.46C58.42 87.96 56.77 88.4 55.1 88.75C53.43 89.09 51.72 89.37 50 89.54C48.28 89.71 46.52 89.81 44.76 89.76C43.01 89.72 41.22 89.58 39.47 89.29C37.72 89 35.95 88.58 34.26 88C32.57 87.42 30.89 86.69 29.33 85.8C27.77 84.92 26.26 83.87 24.91 82.7C23.56 81.53 22.3 80.19 21.23 78.77C20.16 77.35 19.22 75.79 18.47 74.2C17.71 72.61 17.13 70.91 16.7 69.23C16.26 67.55 16.02 65.81 15.88 64.13C15.75 62.45 15.78 60.76 15.88 59.14C15.97 57.52 16.21 55.94 16.45 54.42C16.7 52.89 17.03 51.43 17.34 50C17.65 48.57 18 47.2 18.32 45.83C18.63 44.46 18.94 43.13 19.24 41.76C19.53 40.39 19.79 39.03 20.08 37.61C20.36 36.18 20.62 34.74 20.95 33.23C21.28 31.72 21.61 30.16 22.07 28.57C22.52 26.98 23.02 25.32 23.69 23.69C24.35 22.06 25.11 20.38 26.05 18.79C26.99 17.21 28.08 15.62 29.33 14.2C30.58 12.78 32 11.41 33.54 10.26C35.08 9.12 36.8 8.11 38.57 7.35C40.34 6.6 42.27 6.04 44.17 5.73C46.08 5.43 48.08 5.37 50 5.54C51.92 5.71 53.87 6.15 55.69 6.75C57.52 7.36 59.3 8.22 60.94 9.19C62.57 10.15 64.11 11.33 65.51 12.56C66.91 13.78 68.17 15.15 69.33 16.52C70.48 17.89 71.5 19.34 72.44 20.76C73.38 22.18 74.19 23.62 74.97 25.03C75.75 26.43 76.44 27.83 77.12 29.19C77.8 30.56 78.42 31.89 79.05 33.23C79.67 34.57 80.28 35.87 80.87 37.21C81.46 38.55 82.05 39.88 82.6 41.27C83.14 42.65 83.68 44.05 84.14 45.51C84.6 46.96 85.03 48.46 85.34 50Z"
}, We = {
  droid: "M47.5 14H52.5V32H47.5ZM43 11A7 7 0 1 1 57 11A7 7 0 1 1 43 11Z",
  mech: "M19.5 32L24.5 32L17 13L12 13ZM75.5 32L80.5 32L88 13L83 13ZM10 11.5A4.5 4.5 0 1 1 19 11.5A4.5 4.5 0 1 1 10 11.5ZM81 11.5A4.5 4.5 0 1 1 90 11.5A4.5 4.5 0 1 1 81 11.5Z"
};
function Ie(t) {
  const e = t.trim(), s = e.match(/^hsla?\(\s*([\d.]+)(?:deg)?[,\s]+([\d.]+)%[,\s]+([\d.]+)%/i);
  if (s) return Gs([Number(s[1]) / 360, Number(s[2]) / 100, Number(s[3]) / 100]);
  const a = e.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (a) {
    let i = a[1];
    i.length === 3 && (i = i.split("").map((c) => c + c).join(""));
    const r = parseInt(i, 16);
    return [r >> 16 & 255, r >> 8 & 255, r & 255];
  }
  const o = e.match(/^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/i);
  return o ? [Number(o[1]), Number(o[2]), Number(o[3])] : null;
}
function Zs(t) {
  const e = Ie(t);
  if (!e) return 0.5;
  const s = (a) => {
    const o = a / 255;
    return o <= 0.03928 ? o / 12.92 : ((o + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * s(e[0]) + 0.7152 * s(e[1]) + 0.0722 * s(e[2]);
}
const Os = "#1E1A33", Hs = "#F7F5F2";
function Vs(t) {
  return Zs(t) < 0.13 ? Hs : Os;
}
function Ks([t, e, s]) {
  t /= 255, e /= 255, s /= 255;
  const a = Math.max(t, e, s), o = Math.min(t, e, s), i = (a + o) / 2;
  if (a === o) return [0, 0, i];
  const r = a - o, c = i > 0.5 ? r / (2 - a - o) : r / (a + o);
  let n = 0;
  return a === t ? n = (e - s) / r + (e < s ? 6 : 0) : a === e ? n = (s - t) / r + 2 : n = (t - e) / r + 4, [n / 6, c, i];
}
function Gs([t, e, s]) {
  if (e === 0) return [s * 255, s * 255, s * 255];
  const a = s < 0.5 ? s * (1 + e) : s + e - s * e, o = 2 * s - a, i = (r) => (r = (r % 1 + 1) % 1, r < 1 / 6 ? o + (a - o) * 6 * r : r < 1 / 2 ? a : r < 2 / 3 ? o + (a - o) * (2 / 3 - r) * 6 : o);
  return [i(t + 1 / 3) * 255, i(t) * 255, i(t - 1 / 3) * 255];
}
function Bs([t, e, s]) {
  return `hsl(${(t * 360).toFixed(1)} ${(e * 100).toFixed(1)}% ${(s * 100).toFixed(1)}%)`;
}
const _e = (t) => Math.min(1, Math.max(0, t));
function Ct(t, e, s = 0) {
  const a = Ie(t);
  if (!a) return t;
  const [o, i, r] = Ks(a);
  return Bs([o, _e(i + s + (e < 0 ? -e * 0.25 : 0)), _e(r + e)]);
}
const ne = ["default", "working", "sleeping"], ie = 0.68, gs = 26, zs = 0.2, De = { height: gs, time: ie, stretch: 1, squash: 1.15, squashTime: 0.37, squashEase: "pulse", groundTime: 0.11, groundEase: "pulse", riseTime: 0.33, riseEase: "pulse", clickSquashTime: 0.24, spin: 1, lean: 6, every: 8, land: 0 }, As = (t, e) => e ? t.clickSquashTime : zs * t.time, Ce = (t, e) => As(t, e) + t.time + ws[t.squashEase] * (e ? t.clickSquashTime : t.squashTime) + Math.max(0, t.groundTime) + t.riseTime + Math.max(0, t.land) + 0.05, ws = { sharp: 0, pulse: 2 / 7, soft: 0.5, bouncy: 0.144 }, Ze = (t, e) => {
  if (t <= 0) return 1;
  if (t >= 1) return 0;
  switch (e) {
    case "sharp":
      return (1 - t) * (1 - t);
    case "soft":
      return 0.5 + 0.5 * Math.cos(Math.PI * t);
    case "bouncy":
      return Math.exp(-3.2 * t) * Math.cos(5.4 * t) - t * t * t * 0.026;
    default: {
      const s = 4.2 * t;
      return (1 + s) * Math.exp(-s) - t * t * t * 0.078;
    }
  }
}, Oe = (t, e) => 1 + 0.25 * bs(t, e), bs = (t, e) => {
  if (t <= 0 || t >= 1) return 0;
  let s;
  switch (e) {
    case "sharp":
      s = (1 - t) * (1 - t);
      break;
    case "soft":
      s = Math.sin(Math.PI * t) ** 2;
      break;
    case "bouncy":
      s = Math.exp(-3.15 * t) * Math.sin(8.43 * t) / 0.596;
      break;
    default: {
      const o = 7 * t;
      s = o * o * Math.exp(2 - o) / 4;
    }
  }
  const a = t > 0.85 ? 1 - (t - 0.85) / 0.15 : 1;
  return s * a * a * (3 - 2 * a);
}, ye = (t) => Math.exp(-Math.pow(Math.min(Math.abs(t), Math.abs(t - 1)) / 0.11, 2)), rt = Math.PI / 180, vt = Math.PI * 2, js = { default: 1.2, working: 0.7, sleeping: 1.4 }, Us = 1;
function Qs(t) {
  let e = t * 2654435761 >>> 0 || 1;
  return () => {
    e = e + 1831565813 >>> 0;
    let s = e;
    return s = Math.imul(s ^ s >>> 15, s | 1), s ^= s + Math.imul(s ^ s >>> 7, s | 61), ((s ^ s >>> 14) >>> 0) / 4294967296;
  };
}
function $t(t, e, s, a) {
  return t + (e - t) * (1 - Math.exp(-s * a));
}
const Ht = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2, Js = (t) => 0.5 - 0.5 * Math.cos(Math.PI * t);
class Vt {
  /* A channel that picks a new target now and then and moves to it. The
     head channels move as a lightly damped spring — the turn starts and
     ends softly, the way a head does — while the eyes dart with an
     exponential approach, the way eyes do. */
  constructor(e, s, a, o, i, r = !1) {
    this.rand = e, this.amp = s, this.holdMin = a, this.holdMax = o, this.rate = i, this.spring = r, this.value = 0, this.vel = 0, this.target = 0, this.next = 0;
  }
  update(e, s) {
    if (e >= this.next && (this.target = (this.rand() * 2 - 1) * this.amp, this.next = e + this.holdMin + this.rand() * (this.holdMax - this.holdMin)), this.spring) {
      const a = this.rate * 1.6, o = 0.9;
      this.vel += (a * a * (this.target - this.value) - 2 * o * a * this.vel) * s, this.value += this.vel * s;
    } else this.value = $t(this.value, this.target, this.rate, s);
  }
  /** aim at a value and stay there: the rig picks the next one */
  aim(e) {
    this.target = e, this.next = 1 / 0;
  }
  set(e, s, a, o) {
    this.amp = e, this.holdMin = s, this.holdMax = a, this.rate = o, this.next = 0;
  }
}
class Kt {
  constructor(e) {
    this.duration = e, this.p = -1;
  }
  fire() {
    this.p = 0;
  }
  get active() {
    return this.p >= 0;
  }
  update(e) {
    this.p < 0 || (this.p += e / this.duration, this.p >= 1 && (this.p = -1));
  }
}
const He = 35 * rt, Ve = 14 * rt, Ke = 3.2 * rt, Ge = 2.6, Ns = 4.4, ks = {
  default: { pitch: 0, roll: 0, y: 0, lookX: 0, lookY: 0 },
  working: { pitch: 5 * rt, roll: 0, y: 0, lookX: 0, lookY: 0 },
  sleeping: { pitch: -16 * rt, roll: 6 * rt, y: 3, lookX: 0, lookY: 1 }
};
class ta {
  constructor(e, s = "default") {
    this.pose = { yaw: 0, pitch: 0, roll: 0, x: 0, y: 0, sx: 1, sy: 1, eyeOpen: 1, blinkL: 0, blinkR: 0, lookX: 0, lookY: 0, breath: 0, laugh: 0, whirl: 0, whirlAngle: 0, w: [1, 0, 0] }, this.state = "default", this.t = 0, this.wFrom = [1, 0, 0], this.tr = 1, this.trDuration = 1.2, this.blink = new Kt(0.17), this.blinkAgain = !1, this.dart = new Kt(0.12), this.dartX = 0, this.dartY = 0, this.flip = new Kt(Ce(De, !1)), this.flipPoked = !1, this.jump = { ...De }, this.flipSide = 1, this.nod = new Kt(1.7), this.hopPhase = 0, this.hopCount = 0, this.hopGain = 0, this.laughEv = new Kt(0.8), this.prevYaw = 0, this.jelly = 0, this.jellyV = 0, this.gazeLead = 0, this.gazeDir = [0, 0], this.gazeAt = 0, this.turnK = 1, this.breathPhase = 0, this.ptrX = 0, this.ptrY = 0, this.ptrS = 0, this.ptrTargetX = 0, this.ptrTargetY = 0, this.ptrTargetS = 0, this.baseYaw = 0, this.rand = Qs(Math.floor(e * 1e6) + 1);
    const a = this.rand;
    this.yawW = new Vt(a, 36 * rt, 1.1, 2.6, 3, !0), this.pitchW = new Vt(a, 10 * rt, 1.1, 2.6, 2.6, !0), this.rollW = new Vt(a, 5 * rt, 1.6, 3.2, 2, !0), this.lookXW = new Vt(a, 3.6, 0.5, 2, 14), this.lookYW = new Vt(a, 2.4, 0.5, 2, 14), this.t = a() * 10, this.hopPhase = a(), this.breathPhase = a(), this.blinkAt = this.t + 1 + a() * 3, this.flipAt = this.nextFlip(this.t, 1), this.nodAt = this.t + 3 + a() * 4, this.dartAt = this.t + 1 + a() * 2, this.laughAt = this.t + 0.6 + a() * 1.5, this.setState(s, !0);
  }
  setState(e, s = !1) {
    if (e === this.state && !s) return;
    const a = this.state;
    this.state = e;
    const o = this.pose.w;
    if (s) {
      for (let i = 0; i < 3; i++) o[i] = ne[i] === e ? 1 : 0;
      this.tr = 1;
    } else
      this.wFrom = [o[0], o[1], o[2]], this.tr = 0, this.trDuration = a === "sleeping" ? Us : js[e];
    switch (e) {
      case "default":
        this.yawW.set(He, 2.6, 5.4, 2), this.pitchW.set(Ve, 2.8, 5.8, 1.8), this.rollW.set(Ke, 3.4, 6.6, 1.5), this.gazeAt = 0, this.gazeDir = [0, 0], this.lookXW.set(3.6, 0.6, 2.2, 13), this.lookYW.set(2.4, 0.6, 2.2, 13), this.flipAt = this.nextFlip(this.t, 0.6);
        break;
      case "working":
        this.yawW.set(16 * rt, 0.9, 1.8, 4), this.pitchW.set(3 * rt, 1.2, 2.4, 3), this.rollW.set(0, 1, 2, 3), this.lookXW.set(2, 0.5, 1.2, 12), this.lookYW.set(1, 0.5, 1.2, 12), this.hopPhase = 0, this.hopCount = 0, this.laughAt = this.t + 0.5 + this.rand() * 1.2;
        break;
      case "sleeping":
        this.yawW.set(7 * rt, 3, 6, 0.7), this.pitchW.set(3 * rt, 3, 6, 0.7), this.rollW.set(2 * rt, 3, 6, 0.6), this.lookXW.set(0, 2, 4, 2), this.lookYW.set(0, 2, 4, 2), this.nodAt = this.t + 2.5 + this.rand() * 4;
        break;
    }
  }
  /** Where the pointer is, relative to the head (−1 … 1 across a head
      width), and how strongly to follow it (0 lets go). */
  setPointer(e, s, a) {
    this.ptrTargetX = Math.max(-1.2, Math.min(1.2, e)), this.ptrTargetY = Math.max(-1.2, Math.min(1.2, s)), this.ptrTargetS = Math.max(0, Math.min(1, a));
  }
  /** A hop and a full turn, right now, whatever the state. */
  poke() {
    this.flip.active && this.flip.p < 0.6 || (this.flipPoked = !0, this.flip.duration = Ce(this.jump, !0), this.flipSide = this.rand() < 0.5 ? -1 : 1, this.flip.fire(), this.flipAt = this.nextFlip(this.t, 1.1));
  }
  /** How far the head turns to the side while idle: 1 as the gaze has
      it, 0 keeps it facing forward. */
  setTurn(e) {
    const s = Math.max(0, e);
    s !== this.turnK && (this.turnK = s, this.state === "default" && (this.gazeAt = 0));
  }
  /** The jump's numbers; any subset. */
  setJump(e) {
    const s = this.jump.every;
    Object.assign(this.jump, e), e.every !== void 0 && e.every !== s && (this.flipAt = this.nextFlip(this.t, 1));
  }
  /* Where the head looks next. From a corner it mostly swings straight
     across to the opposite one — top right, stay, bottom left — now and
     then only sideways, or back to the middle for a beat. */
  nextGaze() {
    const e = this.rand, [s, a] = this.gazeDir;
    if (s !== 0 || a !== 0) {
      const i = e();
      return i < 0.66 ? [-s, -a] : i < 0.85 ? [-s, a] : [0, 0];
    }
    const o = [
      [1, -1],
      [-1, 1],
      [-1, -1],
      [1, 1]
    ];
    return o[Math.floor(e() * o.length)];
  }
  /** when the next idle jump is due: `every` seconds, give or take 40 % */
  nextFlip(e, s) {
    const a = this.jump.every;
    return a > 0 ? e + a * s * (0.625 + this.rand() * 0.75) : 1 / 0;
  }
  /** Advance by `dt` seconds (already scaled by the speed). */
  update(e) {
    e = Math.min(e, 0.05), this.t += e;
    const s = this.t, a = this.pose, o = a.w;
    if (this.tr < 1) {
      this.tr = Math.min(1, this.tr + e / this.trDuration);
      const f = Js(this.tr);
      for (let b = 0; b < 3; b++) {
        const R = ne[b] === this.state ? 1 : 0;
        o[b] = this.wFrom[b] + (R - this.wFrom[b]) * f;
      }
    }
    const [i, r, c] = o, n = { pitch: 0, roll: 0, y: 0, lookX: 0, lookY: 0 };
    for (let f = 0; f < 3; f++) {
      const b = ks[ne[f]];
      n.pitch += b.pitch * o[f], n.roll += b.roll * o[f], n.y += b.y * o[f], n.lookX += b.lookX * o[f], n.lookY += b.lookY * o[f];
    }
    if (this.state === "default" && s >= this.gazeAt) {
      const [f, b] = this.nextGaze();
      this.gazeDir = [f, b];
      const R = 0.84 + this.rand() * 0.16;
      this.yawW.aim(f * He * R * this.turnK), this.pitchW.aim(b * Ve * R), this.rollW.aim(f * Ke * R * this.turnK), this.gazeAt = s + Ge + this.rand() * (Ns - Ge);
    }
    this.yawW.update(s, e), this.pitchW.update(s, e), this.rollW.update(s, e), this.lookXW.update(s, e), this.lookYW.update(s, e), this.ptrS = $t(this.ptrS, this.ptrTargetS, 8, e), this.ptrX = $t(this.ptrX, this.ptrTargetX, 14, e), this.ptrY = $t(this.ptrY, this.ptrTargetY, 14, e);
    const l = this.ptrS, h = 1 - 0.75 * l;
    this.baseYaw = $t(this.baseYaw, this.yawW.value * h + 22 * rt * this.ptrX * l, 5, e);
    const d = n.pitch + this.pitchW.value * h - 12 * rt * this.ptrY * l, M = n.roll + this.rollW.value * h, p = n.y, u = n.lookX + this.lookXW.value * h + 4.5 * this.ptrX * l, A = n.lookY + this.lookYW.value * h + 3 * this.ptrY * l;
    let m = 0, T = 0, E = 1, X = 1, g = 0, F = 0, x = 0, _ = 0, w = 0, C = 0, I = 0, k = 0;
    const y = (f, b, R) => {
      const Y = Math.min(1, Math.max(0, (R - f) / (b - f)));
      return Y * Y * (3 - 2 * Y);
    }, P = (f) => y(0.1, 0.26, f) * (1 - y(0.66, 0.9, f)), L = (f) => vt * (1.5 * f + 0.9 * Ht(f));
    if (s >= this.blinkAt && !this.blink.active && i + r > 0.5 && (this.blink.fire(), this.blinkAgain = !this.blinkAgain && this.rand() < 0.22, this.blinkAt = s + (this.blinkAgain ? 0.28 : 2.2 + this.rand() * 2.6)), this.blink.update(e), this.blink.active && (x = Math.sin(Math.PI * this.blink.p)), s >= this.dartAt && !this.dart.active && i + r > 0.5 && (this.dart.fire(), this.dartX = (this.rand() * 2 - 1) * 4, this.dartY = (this.rand() * 2 - 1) * 2, this.dart.duration = 0.25 + this.rand() * 0.45, this.dartAt = s + 1.2 + this.rand() * 2.6), this.dart.update(e), this.dart.active) {
      const f = this.dart.p, b = f < 0.15 ? f / 0.15 : f > 0.8 ? (1 - f) / 0.2 : 1;
      _ += this.dartX * b * (i + r), w += this.dartY * b * (i + r);
    }
    if (this.state === "default" && s >= this.flipAt && !this.flip.active && (this.flipPoked = !1, this.flip.duration = Ce(this.jump, !1), this.flipSide = this.rand() < 0.5 ? -1 : 1, this.flip.fire(), this.flipAt = this.nextFlip(s, 1)), this.flip.update(e), this.flip.active) {
      const f = this.jump, b = As(f, this.flipPoked), R = (this.flip.p * this.flip.duration - b) / f.time, Y = Math.min(1, Math.max(0, R)), S = Math.sin(Math.PI * Y);
      m += vt * f.spin * Ht(Y), T -= f.height * S;
      const v = (R - 1) * f.time - f.land, $ = this.flipPoked ? f.clickSquashTime : f.squashTime, q = (it) => this.flipPoked ? it * it * (3 - 2 * it) : ye(it - 1), Z = Math.max(0, f.groundTime), G = ws[f.squashEase] * $, et = v - G - Z, nt = v <= G ? bs(v / $, f.squashEase) : et <= 0 ? Oe((v - G) / Z, f.groundEase) : Ze(et / f.riseTime, f.riseEase), ft = (R < 0 ? q(Math.max(0, 1 + R * f.time / b)) : v > 0 ? nt : R < 0.2 ? ye(R) : 0) * f.squash;
      E += 0.16 * ft - 0.06 * S * f.stretch, X += -0.18 * ft + 0.09 * S * f.stretch, F += this.flipSide * f.lean * rt * S, f.spin > 0 && (C = Math.max(C, S), I = Math.max(I, P(Y)), k = L(Y));
    }
    const B = this.jump, j = Math.max(0, B.groundTime), ct = j + B.riseTime;
    if (this.state === "working" && (this.hopGain = r), this.hopGain > 0.02 && (this.state === "working" || this.hopPhase > 0)) {
      this.hopPhase += e / ie, this.hopPhase >= 1 && (this.state === "working" ? (this.hopPhase -= 1, this.hopCount += 1) : (this.hopPhase - 1) * ie >= ct && (this.hopPhase = 0, this.hopGain = 0));
      const f = this.hopGain, b = Math.min(1, this.hopPhase), R = Math.sin(Math.PI * b), Y = this.hopCount % 3 === 2;
      T -= (Y ? gs : 18) * R * f;
      const v = this.state !== "working" && this.hopPhase > 1 ? (this.hopPhase - 1) * ie : -1, $ = v < 0 ? ye(this.hopPhase) : v < j ? Oe(v / j, B.groundEase) : Ze((v - j) / B.riseTime, B.riseEase);
      E += (0.16 * $ - 0.06 * R) * f, X += (-0.18 * $ + 0.09 * R) * f, Y && (m += vt * Ht(b) * f, C = Math.max(C, R * f), P(b) * f > I && (I = P(b) * f, k = L(b))), F += (this.hopCount % 2 === 0 ? 1 : -1) * 6 * rt * R * f;
    }
    if (this.state === "working" && s >= this.laughAt && !this.laughEv.active && (this.laughEv.fire(), this.laughEv.duration = 0.6 + this.rand() * 0.5, this.laughAt = s + 1.6 + this.rand() * 2.2), this.laughEv.update(e), this.laughEv.active) {
      const f = this.laughEv.p;
      C = Math.max(C, f < 0.18 ? f / 0.18 : f > 0.78 ? (1 - f) / 0.22 : 1);
    }
    if (this.state === "sleeping" && s >= this.nodAt && !this.nod.active && (this.nod.fire(), this.nodAt = s + 4 + this.rand() * 4), this.nod.update(e), this.nod.active) {
      const f = this.nod.p, b = f < 0.72 ? Ht(f / 0.72) : 1 - Ht((f - 0.72) / 0.28);
      g -= 13 * rt * b * c;
    }
    this.breathPhase += e / (3.6 + 1.2 * c);
    const W = Math.sin(this.breathPhase * vt);
    a.breath = W, E += W * (8e-3 + 0.014 * c), X += W * (0.012 + 0.02 * c);
    const V = Math.sin(s * vt / 3.4) * 2 * (1 - c);
    a.yaw = this.baseYaw + m;
    let D = this.baseYaw - this.prevYaw;
    D = ((D + Math.PI) % vt + vt) % vt - Math.PI, this.prevYaw = this.baseYaw;
    const z = e > 0 ? Math.abs(D) / e : 0, U = e > 0 ? Math.max(-2.2, Math.min(2.2, D / e * 2.4)) : 0;
    this.gazeLead = $t(this.gazeLead, U, 9, e);
    const Q = Math.min(0.22, 0.055 * z), st = 16, tt = 0.45;
    this.jellyV += (st * st * (Q - this.jelly) - 2 * tt * st * this.jellyV) * e, this.jelly += this.jellyV * e;
    const J = Math.max(-0.08, Math.min(0.28, this.jelly)) * 0.6;
    E *= 1 + J, X *= 1 - 0.55 * J, a.pitch = d + g, a.roll = M + F, a.x = 0, a.y = p + T + V, a.sx = E, a.sy = X, a.eyeOpen = 1, a.laugh = $t(a.laugh, C, 30, e), a.blinkL = x, a.blinkR = x, a.lookX = u + _ + this.gazeLead, a.lookY = A + w, a.whirl = I, a.whirlAngle = k;
  }
}
function Be(t) {
  const e = ks[t];
  return {
    yaw: 0,
    pitch: e.pitch,
    roll: e.roll,
    x: 0,
    y: e.y,
    sx: 1,
    sy: 1,
    eyeOpen: 1,
    blinkL: 0,
    blinkR: 0,
    lookX: e.lookX,
    lookY: e.lookY,
    breath: 0,
    laugh: 0,
    whirl: 0,
    whirlAngle: 0,
    w: ne.map((s) => s === t ? 1 : 0)
  };
}
const xt = 3, Ft = 100 + 2 * xt, at = 64, Za = at, oe = at * at, ge = 24, ea = 1e12, xs = 48 * Math.PI / 180, ze = Math.cos(xs), sa = Math.sin(xs), Ae = 17, aa = (t, e) => e + (1 - e) * Math.sqrt(Math.max(0, 1 - t * t));
function je(t, e, s, a, o, i) {
  let r = 0;
  o[0] = 0, i[0] = -1e30, i[1] = 1e30;
  for (let c = 1; c < e; c++) {
    let n = 0;
    for (; ; ) {
      const l = o[r];
      if (n = (t[c] + c * c - t[l] - l * l) / (2 * (c - l)), n > i[r]) break;
      r--;
    }
    r++, o[r] = c, i[r] = n, i[r + 1] = 1e30;
  }
  r = 0;
  for (let c = 0; c < e; c++) {
    for (; i[r + 1] < c; ) r++;
    const n = o[r];
    s[c] = (c - n) * (c - n) + t[n], a[c] = n;
  }
}
function Ue(t, e, s, a, o) {
  const i = new Float32Array(s), r = new Float32Array(s), c = new Int32Array(s), n = new Int32Array(s), l = new Float32Array(s + 1), h = new Float32Array(s * s), d = new Int32Array(s * s);
  for (let M = 0; M < s; M++) {
    for (let p = 0; p < s; p++) i[p] = t[p * s + M] === e ? 0 : ea;
    je(i, s, r, c, n, l);
    for (let p = 0; p < s; p++)
      h[p * s + M] = r[p], d[p * s + M] = c[p];
  }
  for (let M = 0; M < s; M++) {
    const p = M * s;
    for (let u = 0; u < s; u++) i[u] = h[p + u];
    je(i, s, r, c, n, l);
    for (let u = 0; u < s; u++)
      a[p + u] = r[u], o && (o[p + u] = d[p + c[u]] * s + c[u]);
  }
}
function we(t, e, s) {
  for (let a = 0; a < e; a++) {
    const o = a * e;
    for (let i = 0; i < e; i++) {
      const r = i < 2 ? 0 : i - 2, c = i < 1 ? 0 : i - 1, n = i > e - 2 ? e - 1 : i + 1, l = i > e - 3 ? e - 1 : i + 2;
      s[o + i] = (t[o + r] + 4 * t[o + c] + 6 * t[o + i] + 4 * t[o + n] + t[o + l]) * 0.0625;
    }
  }
  for (let a = 0; a < e; a++)
    for (let o = 0; o < e; o++) {
      const i = o < 2 ? 0 : o - 2, r = o < 1 ? 0 : o - 1, c = o > e - 2 ? e - 1 : o + 1, n = o > e - 3 ? e - 1 : o + 2;
      t[o * e + a] = (s[i * e + a] + 4 * s[r * e + a] + 6 * s[o * e + a] + 4 * s[c * e + a] + s[n * e + a]) * 0.0625;
    }
}
function na(t, e, s) {
  const a = [];
  for (let i = e, r = 1; r <= 4 && i % 2 === 0 || r === 1; i >>= 1, r <<= 1) {
    const c = new Uint8Array(i * i);
    for (let n = 0; n < i; n++)
      for (let l = 0; l < i; l++) {
        let h = 0;
        for (let d = 0; d < r; d++) for (let M = 0; M < r; M++) h += t[(n * r + d) * e + l * r + M];
        c[n * i + l] = h >= 128 * r * r ? 1 : 0;
      }
    if (a.push({ n: i, mask: c, phi: new Float32Array(i * i) }), r === 4) break;
  }
  const o = (i, r, c, n) => {
    const { n: l, mask: h, phi: d } = i, M = r * r;
    for (let p = 0; p < c; p++)
      for (let u = 1; u < l - 1; u++) {
        const A = u * l;
        for (let m = 1; m < l - 1; m++) {
          const T = A + m;
          if (!h[T]) continue;
          const E = (d[T - 1] + d[T + 1] + d[T - l] + d[T + l] + M) * 0.25;
          d[T] += n * (E - d[T]);
        }
      }
  };
  for (let i = a.length - 1; i >= 0; i--) {
    const r = a[i], c = 1 << i;
    if (i < a.length - 1) {
      const l = a[i + 1], h = r.n, d = l.n;
      for (let M = 0; M < h; M++) {
        const p = Math.min(d - 1, Math.max(0, (M + 0.5) / 2 - 0.5)), u = p | 0, A = Math.min(d - 1, u + 1), m = p - u;
        for (let T = 0; T < h; T++) {
          const E = M * h + T;
          if (!r.mask[E]) continue;
          const X = Math.min(d - 1, Math.max(0, (T + 0.5) / 2 - 0.5)), g = X | 0, F = Math.min(d - 1, g + 1), x = X - g;
          r.phi[E] = (l.phi[u * d + g] * (1 - x) + l.phi[u * d + F] * x) * (1 - m) + (l.phi[A * d + g] * (1 - x) + l.phi[A * d + F] * x) * m;
        }
      }
    }
    const n = Math.min(1.9, 2 / (1 + Math.sin(Math.PI / r.n)) - 0.05);
    o(r, s * c, 4, 1), o(r, s * c, i === 2 ? 100 : i === 1 ? 30 : 16, n), o(r, s * c, 8, 1);
  }
  return a[0].phi;
}
const ia = [1, 1, 0, -1, -1, -1, 0, 1], oa = [0, 1, 1, 1, 0, -1, -1, -1], ra = [1, Math.SQRT2, 1, Math.SQRT2, 1, Math.SQRT2, 1, Math.SQRT2];
function ca(t, e, s) {
  const a = Ft / e, o = e * e, i = new Uint8Array(o);
  for (let C = 0; C < o; C++) i[C] = t[C] >= 128 ? 1 : 0;
  const r = new Float32Array(o), c = new Float32Array(o), n = new Int32Array(o);
  Ue(i, 0, e, r, null), Ue(i, 1, e, c, n);
  const l = new Float32Array(o), h = new Float32Array(o);
  for (let C = 0; C < o; C++) {
    const I = t[C] / 255;
    l[C] = a * (I > 0 && I < 1 ? I - 0.5 : i[C] ? Math.sqrt(r[C]) - 0.5 : 0.5 - Math.sqrt(c[C]));
  }
  we(l, e, h), we(l, e, h);
  const d = na(t, e, a);
  let M = 0;
  for (let C = 0; C < o; C++) d[C] > M && (M = d[C]);
  const p = 2 * Math.sqrt(M), u = Math.min(0.9 * s + 0.12 * p, 1.2 * p), A = M > 0 ? u / Math.sqrt(M) : 0, m = new Float32Array(o);
  for (let C = 0; C < o; C++) m[C] = d[C] > 0 ? A * Math.sqrt(d[C]) : 0;
  we(m, e, h);
  const T = { N: e, i00: new Uint16Array(o), wx: new Uint8Array(o), wy: new Uint8Array(o), ao: new Uint8Array(o) }, { i00: E, wx: X, wy: g, ao: F } = T, x = e <= 64 ? [1, 2, 3, 5, 8] : e <= 96 ? [1, 2, 4, 7, 11] : [1, 2, 4, 7, 11, 15], _ = 9, w = e - 1;
  for (let C = 0; C < e; C++)
    for (let I = 0; I < e; I++) {
      const k = C * e + I, y = i[k] ? k : c[k] <= _ ? n[k] : -1;
      if (y < 0) continue;
      const P = y % e, L = (y - P) / e, B = P > 0 ? P - 1 : 0, j = P < w ? P + 1 : w, ct = L > 0 ? L - 1 : 0, W = L < w ? L + 1 : w;
      let V = -(m[L * e + j] - m[L * e + B]) / (2 * a), D = -(m[W * e + P] - m[ct * e + P]) / (2 * a), z = 1, U = Math.sqrt(V * V + D * D + 1);
      V /= U, D /= U, z /= U;
      const Q = Math.max(0, l[y]);
      if (Q < 2) {
        let $ = l[L * e + j] - l[L * e + B], q = l[W * e + P] - l[ct * e + P];
        const Z = Math.hypot($, q) || 1;
        $ /= Z, q /= Z;
        const G = 0.7 * (1 - Q / 2);
        V += G * (-$ - V), D += G * (-q - D), z += G * (0 - z), U = Math.sqrt(V * V + D * D + z * z) || 1, V /= U, D /= U, z /= U;
      }
      const st = m[y];
      let tt = 0;
      for (let $ = 0; $ < 8; $++) {
        let q = 0;
        for (let Z = 0; Z < x.length; Z++) {
          const G = x[Z];
          let et = P + ia[$] * G, nt = L + oa[$] * G;
          et < 0 ? et = 0 : et > w && (et = w), nt < 0 ? nt = 0 : nt > w && (nt = w);
          const ft = (m[nt * e + et] - st) / (G * a * ra[$]);
          ft > q && (q = ft);
        }
        tt += q / Math.sqrt(1 + q * q);
      }
      const J = 1 - Math.min(1, Q / 3), f = 1 - 0.2 * J * J, b = Math.pow(1 - 0.9 * tt / 8, 1.5) * f;
      F[k] = Math.max(1, Math.round(255 * Math.pow(b, 1 / 2.2)));
      const R = (V * 0.5 + 0.5) * (at - 1), Y = (D * 0.5 + 0.5) * (at - 1), S = Math.min(at - 2, Math.max(0, R | 0)), v = Math.min(at - 2, Math.max(0, Y | 0));
      E[k] = v * at + S, X[k] = Math.round(255 * Math.min(1, Math.max(0, R - S))), g[k] = Math.round(255 * Math.min(1, Math.max(0, Y - v)));
    }
  return T;
}
function fe(t) {
  if (typeof OffscreenCanvas == "function") return new OffscreenCanvas(t, t);
  if (typeof document < "u") {
    const e = document.createElement("canvas");
    return e.width = e.height = t, e;
  }
  return null;
}
function ue(t, e) {
  return t.getContext("2d", e ? { willReadFrequently: !0 } : void 0);
}
function ha(t, e) {
  const s = fe(e), a = s && ue(s, !0);
  if (!a) return null;
  const o = Ft / e;
  a.setTransform(1 / o, 0, 0, 1 / o, xt / o, xt / o), a.fillStyle = "#fff", a.fill(t);
  const i = a.getImageData(0, 0, e, e).data, r = new Uint8ClampedArray(e * e);
  for (let c = 0; c < e * e; c++) r[c] = i[c * 4 + 3];
  return r;
}
const qt = /* @__PURE__ */ new Map(), be = /* @__PURE__ */ new Set(), Qe = /* @__PURE__ */ new WeakMap();
let la = 0;
function Je(t) {
  let e = Qe.get(t);
  return e || Qe.set(t, e = `p${la++}`), e;
}
const Fe = [];
let Le = !1;
function Ne() {
  Le = !1;
  const t = Fe.shift();
  t && t(), Fe.length && Ss();
}
function Ss() {
  if (Le) return;
  Le = !0;
  const t = globalThis.requestIdleCallback;
  t ? t(Ne, { timeout: 120 }) : setTimeout(Ne, 16);
}
function fa(t) {
  Fe.push(t), Ss();
}
function Ts(t, e, s, a, o) {
  const i = `${t}|${s}|${Math.round(a)}`, r = qt.get(i);
  if (r) return r;
  const c = () => {
    if (be.delete(i), qt.has(i)) return;
    const n = ha(e, s);
    n && (qt.size >= 48 && qt.clear(), qt.set(i, ca(n, s, a)));
  };
  return o ? (c(), qt.get(i) ?? null) : (be.has(i) || (be.add(i), fa(c)), null);
}
function ua(t, e, s = 192, a = 0.65) {
  Ts(t, e, Ps(s), 15 * a, !0);
}
function Ps(t) {
  return t <= 100 ? 64 : t <= 224 ? 96 : 128;
}
const ke = (t) => t <= 0.04045 ? t / 12.92 : Math.pow((t + 0.055) / 1.055, 2.4), zt = /* @__PURE__ */ new Map();
function da(t) {
  let e = zt.get(t);
  if (!e) {
    let s = Ie(t);
    if (!s) {
      const a = fe(1), o = a && ue(a, !0);
      if (o) {
        o.fillStyle = t, o.fillRect(0, 0, 1, 1);
        const i = o.getImageData(0, 0, 1, 1).data;
        s = [i[0], i[1], i[2]];
      } else s = [128, 128, 128];
    }
    e = [ke(s[0] / 255), ke(s[1] / 255), ke(s[2] / 255)], zt.size > 200 && zt.clear(), zt.set(t, e);
  }
  return e;
}
const de = 2048, vs = 2.5, Fs = de / vs, Ls = new Float32Array(de);
for (let t = 0; t < de; t++) {
  const e = (t + 0.5) / Fs, s = e <= 0.75 ? e : 0.75 + 0.25 * (1 - Math.exp(-(e - 0.75) / 0.25));
  Ls[t] = 255 * (s <= 31308e-7 ? 12.92 * s : 1.055 * Math.pow(s, 1 / 2.4) - 0.055);
}
const xe = (t) => Ls[t <= 0 ? 0 : t >= vs ? de - 1 : t * Fs | 0], re = 1024, jt = /* @__PURE__ */ new Map();
function ts(t) {
  let e = jt.get(t);
  if (!e) {
    e = new Float32Array(re + 1);
    for (let s = 0; s <= re; s++) e[s] = Math.pow(s / re, t);
    jt.size > 16 && jt.clear(), jt.set(t, e);
  }
  return e;
}
const Se = [0.92, 0.96, 1], Te = [1, 0.98, 0.95], Es = (t, e, s) => {
  const a = s <= t ? 0 : s >= e ? 1 : (s - t) / (e - t);
  return a * a * (3 - 2 * a);
}, es = (t, e, s) => 1 - Es(t - e, t + e, s);
function pa(t, e, s, a) {
  const { L: o, V: i, H: r, U: c, W: n, A: l, B: h } = s, d = Math.max(e[0], e[1], e[2], 0.05), M = [e[0] / d, e[1] / d, e[2] / d], p = Math.max(0.03, 0.3 - 0.15 * a.shadow), u = 0.15 + 0.14 * a.spread, A = 0.85, m = Math.min(90, Math.round(110 / Math.pow(a.spread, 1.3))), T = Math.max(2, Math.round(8 / a.spread)), E = ts(m), X = ts(T), g = 0.45 * a.highlight, F = 0.1 * a.highlight, x = 0.11 * a.highlight, _ = 0.3 * a.rim, w = [p * M[0], p * M[1], p * M[2]];
  for (let C = 0; C < at; C++)
    for (let I = 0; I < at; I++) {
      let k = I / (at - 1) * 2 - 1, y = C / (at - 1) * 2 - 1, P = k * k + y * y;
      if (P > 1.14) continue;
      if (P > 1) {
        const $ = 1 / Math.sqrt(P);
        k *= $, y *= $, P = 1;
      }
      const L = Math.sqrt(1 - P), B = k * o[0] + y * o[1] + L * o[2], j = Math.max(0, k * i[0] + y * i[1] + L * i[2]), ct = Math.max(0, k * r[0] + y * r[1] + L * r[2]), W = Math.min(1, Math.max(0, (B + u) / (1 + u))), V = 1 - j, D = V * V, z = D * V, U = z * D, Q = ct * re | 0, st = (g * E[Q] + F * X[Q]) * (1 + 3 * U), tt = 2 * j * k - i[0], J = 2 * j * y - i[1], f = 2 * j * L - i[2], b = 0.45 + 0.55 * Es(-0.4, 0.6, tt * c[0] + J * c[1] + f * c[2]), R = tt * n[0] + J * n[1] + f * n[2];
      let Y = 0;
      if (R > 0.5) {
        const $ = (tt * l[0] + J * l[1] + f * l[2]) / R, q = (tt * h[0] + J * h[1] + f * h[2]) / R;
        Y = es(0.34, 0.12, Math.abs($)) * es(0.12, 0.06, Math.abs(q));
      }
      const S = _ * z * b + x * Y, v = (C * at + I) * 3;
      t[v] = xe(e[0] * (w[0] + A * W) + st * Te[0] + S * Se[0]), t[v + 1] = xe(e[1] * (w[1] + A * W) + st * Te[1] + S * Se[1]), t[v + 2] = xe(e[2] * (w[2] + A * W) + st * Te[2] + S * Se[2]);
    }
}
function ss(t, e, s, a) {
  const o = (e * 0.5 + 0.5) * (at - 1), i = (s * 0.5 + 0.5) * (at - 1), r = Math.min(at - 2, Math.max(0, o | 0)), c = Math.min(at - 2, Math.max(0, i | 0)), n = o - r, l = i - c, h = (c * at + r) * 3, d = at * 3, M = (1 - n) * (1 - l), p = n * (1 - l), u = (1 - n) * l, A = n * l;
  for (let m = 0; m < 3; m++) a[m] = t[h + m] * M + t[h + 3 + m] * p + t[h + d + m] * u + t[h + d + 3 + m] * A;
}
function ma(t, e, s, a) {
  const { N: o, i00: i, wx: r, wy: c, ao: n } = t, l = at * 3;
  for (let h = 0, d = 0; h < o * o; h++, d += 4) {
    const M = n[h];
    if (M === 0) {
      s[d + 3] = 0;
      continue;
    }
    const p = a[M], u = i[h] * 3, A = r[h] * (1 / 255), m = c[h] * (1 / 255), T = (1 - A) * (1 - m) * p, E = A * (1 - m) * p, X = (1 - A) * m * p, g = A * m * p;
    s[d] = e[u] * T + e[u + 3] * E + e[u + l] * X + e[u + l + 3] * g, s[d + 1] = e[u + 1] * T + e[u + 4] * E + e[u + l + 1] * X + e[u + l + 4] * g, s[d + 2] = e[u + 2] * T + e[u + 5] * E + e[u + l + 2] * X + e[u + l + 5] * g, s[d + 3] = 255;
  }
}
const Ut = (t) => {
  const e = Math.hypot(t[0], t[1], t[2]) || 1;
  return [t[0] / e, t[1] / e, t[2] / e];
};
function Ma(t) {
  const e = Math.cos(t.roll), s = Math.sin(t.roll), a = e * t.lx + s * t.ly, o = -s * t.lx + e * t.ly, i = t.facing < 0 ? -1 : 1, r = i * Math.min(1, Math.abs(t.facing) / 0.16), { cy: c, sy: n, cp: l, sp: h } = t, d = (I, k, y, P = r) => Ut([c * I + n * h * k - n * l * y, l * k + h * y, P * (n * I - c * h * k + c * l * y)]), M = d(ze * a, ze * o, sa), p = d(0, 0, 1, i), u = Ut([M[0] + p[0], M[1] + p[1], M[2] + p[2]]), A = d(a, o, 0, i), m = Math.cos(80 * Math.PI / 180), T = Math.sin(80 * Math.PI / 180), E = m * a - T * o, X = T * a + m * o, g = Ut([0.55 * E, 0.55 * X, 0.83]), F = Ut([g[1], -g[0], 0]), x = [g[1] * F[2] - g[2] * F[1], g[2] * F[0] - g[0] * F[2], g[0] * F[1] - g[1] * F[0]], _ = d(g[0], g[1], g[2], i), w = d(F[0], F[1], F[2], i), C = d(x[0], x[1], x[2], i);
  return { L: M, V: p, H: u, U: A, W: _, A: w, B: C };
}
const as = /* @__PURE__ */ new WeakMap();
function Ca(t, e) {
  const s = t.canvas ?? t;
  let a = as.get(s);
  a || (a = /* @__PURE__ */ new Map(), as.set(s, a));
  let o = a.get(e);
  return o || (o = {
    N: 0,
    img: null,
    mc: new Float32Array(oe * 3),
    mcPrev: new Float32Array(oe * 3),
    mcMix: new Float32Array(oe * 3),
    mixVersion: 0,
    blendT: 1,
    blendFrames: 1,
    sinceBuild: 0,
    L: null,
    V: null,
    lx: NaN,
    ly: NaN,
    base: "",
    shadow: NaN,
    highlight: NaN,
    spread: NaN,
    rim: NaN,
    version: 0,
    imgVersion: -1,
    imgAoK: NaN,
    imgForm: null,
    aoK: -1,
    aoMul: new Float32Array(256),
    near: null,
    rimG: null,
    far: null,
    scratch: [null, null],
    scratchIdx: 0,
    scratchN: 0,
    scratchStale: !0,
    sprites: [null, null, null],
    spriteVersion: -1,
    spritePx: 0
  }, a.size > 4 && a.clear(), a.set(e, o)), o;
}
const Pe = 1 / 48, ns = (t, e) => !e || Math.abs(t[0] - e[0]) >= Pe || Math.abs(t[1] - e[1]) >= Pe || Math.abs(t[2] - e[2]) >= Pe;
function Ee(t, e) {
  return [
    t[0] * e[0] + t[2] * e[1],
    t[1] * e[0] + t[3] * e[1],
    t[0] * e[2] + t[2] * e[3],
    t[1] * e[2] + t[3] * e[3],
    t[0] * e[4] + t[2] * e[5] + t[4],
    t[1] * e[4] + t[3] * e[5] + t[5]
  ];
}
function Qt(t, e, s, a, o) {
  const i = Math.sqrt(1 - s * s), r = [0, 0, 0], c = 1 - a;
  if (typeof t.createConicGradient == "function") {
    const h = t.createConicGradient(0, 50, 50);
    for (let d = 0; d <= ge; d++) {
      const M = d / ge * Math.PI * 2;
      ss(e, i * Math.cos(M), i * Math.sin(M), r), h.addColorStop(d / ge, `rgb(${r[0] * c | 0} ${r[1] * c | 0} ${r[2] * c | 0})`);
    }
    return h;
  }
  const n = t.createLinearGradient(50 + o[0] * 50, 50 + o[1] * 50, 50 - o[0] * 50, 50 - o[1] * 50), l = (h, d, M) => {
    ss(e, h, d, r), n.addColorStop(M, `rgb(${r[0] * c | 0} ${r[1] * c | 0} ${r[2] * c | 0})`);
  };
  return l(i * o[0], i * o[1], 0), l(-i * o[1], i * o[0], 0.5), l(-i * o[0], -i * o[1], 1), n;
}
const ya = typeof navigator < "u" && /AppleWebKit\//.test(navigator.userAgent) && !/Chrome\/|Chromium\/|Edg\//.test(navigator.userAgent), is = [[0.55, () => 0], [0, () => 0], [0, (t) => Math.min(0.6, 0.25 * t.shadow)]];
function ga(t, e, s, a, o, i) {
  const r = Ps(s.dev), c = Ts(e.typeKey ?? Je(e.path), e.path, r, s.halfDepth, !!s.still);
  if (!c) return !1;
  const n = Ca(t, e.typeKey ?? Je(e.path)), l = Ma(s), h = (() => {
    const S = Math.hypot(l.L[0], l.L[1]);
    return S < 0.05 ? [0, -1] : [l.L[0] / S, l.L[1] / S];
  })();
  if ((ns(l.L, n.L) || ns(l.V, n.V) || s.lx !== n.lx || s.ly !== n.ly || a.base !== n.base || i.shadow !== n.shadow || i.highlight !== n.highlight || i.spread !== n.spread || i.rim !== n.rim) && (n.version > 0 && n.mcPrev.set(n.mcMix), pa(n.mc, da(a.base), l, i), n.version === 0 ? (n.mcMix.set(n.mc), n.blendT = 1) : (n.blendFrames = Math.min(10, Math.max(1, n.sinceBuild)), n.blendT = 0), n.sinceBuild = 0, n.mixVersion++, n.L = l.L, n.V = l.V, n.lx = s.lx, n.ly = s.ly, n.base = a.base, n.shadow = i.shadow, n.highlight = i.highlight, n.spread = i.spread, n.rim = i.rim, n.version++, n.near = n.rimG = n.far = null), n.sinceBuild++, n.blendT < 1) {
    n.blendT = Math.min(1, n.blendT + 1 / n.blendFrames);
    const S = n.blendT >= 1 ? 1 : n.blendT * n.blendT * (3 - 2 * n.blendT), v = n.mcPrev, $ = n.mc, q = n.mcMix;
    for (let Z = 0; Z < oe * 3; Z++) q[Z] = v[Z] + ($[Z] - v[Z]) * S;
    n.mixVersion++;
  }
  const d = Math.min(1.3, 1.2 * i.shadow);
  if (d !== n.aoK) {
    for (let S = 0; S < 256; S++) n.aoMul[S] = Math.max(0, 1 - d * (1 - S / 255));
    n.aoK = d;
  }
  if ((!n.img || n.N !== r) && (n.img = new ImageData(r, r), n.N = r, n.imgVersion = -1), (n.imgVersion !== n.mixVersion || n.imgAoK !== d || n.imgForm !== c) && (ma(c, n.mcMix, n.img.data, n.aoMul), n.imgVersion = n.mixVersion, n.imgAoK = d, n.imgForm = c, n.scratchStale = !0), n.scratchN !== r && (n.scratch = [null, null], n.scratchN = r, n.scratchStale = !0), n.scratchStale) {
    n.scratchIdx ^= 1;
    let S = n.scratch[n.scratchIdx];
    if (!S) {
      const v = fe(r), $ = v && ue(v, !1);
      if (!v || !$) return !1;
      S = n.scratch[n.scratchIdx] = { c: v, g: $ };
    }
    S.g.putImageData(n.img, 0, 0), n.scratchStale = !1;
  }
  const M = n.scratch[n.scratchIdx];
  let p = e.sides === "sprite" || e.sides !== "vector" && ya;
  if (p) {
    const S = Math.ceil(Ft * s.dev / 100);
    if (n.spritePx !== S && (n.sprites = [null, null, null], n.spritePx = S, n.spriteVersion = -1), n.spriteVersion !== n.version) {
      const v = S / Ft;
      for (let $ = 0; $ < 3 && p; $++) {
        let q = n.sprites[$];
        if (!q) {
          const Z = fe(S), G = Z && ue(Z, !1);
          if (!Z || !G) {
            p = !1;
            break;
          }
          q = n.sprites[$] = { c: Z, g: G };
        }
        q.g.setTransform(1, 0, 0, 1, 0, 0), q.g.clearRect(0, 0, S, S), q.g.setTransform(v, 0, 0, v, xt * v, xt * v), q.g.fillStyle = Qt(q.g, n.mc, is[$][0], is[$][1](i), h), q.g.fill(e.path);
      }
      p && (n.spriteVersion = n.version);
    }
  }
  !p && !n.near && (n.near = Qt(t, n.mc, 0.55, 0, h), n.rimG = Qt(t, n.mc, 0, 0, h), n.far = Qt(t, n.mc, 0, Math.min(0.6, 0.25 * i.shadow), h));
  const { cy: u, sy: A, cp: m, sp: T, halfDepth: E, cap: X } = s, g = s.facing >= 0 ? 1 : -1, [F, x, _, w, C, I] = s.ctm;
  let k = null;
  p && (t.imageSmoothingEnabled = !0, t.imageSmoothingQuality = "high");
  let y = 1, P = 0, L = 0, B = 1, j = 0, ct = 0;
  for (let S = 0; S < Ae - 1; S++) {
    const $ = -1 + 2 * (g > 0 ? S : Ae - 1 - S) / (Ae - 1), q = aa($, X), Z = $ * g, G = u * q, et = A * T * q, nt = m * q, ft = $ * A * E - 50 * G, it = -$ * u * T * E - 50 * et - 50 * nt, K = y * B - P * L, ht = B / K, ut = -P / K, lt = -L / K, dt = y / K, ot = (L * ct - B * j) / K, mt = (P * j - y * ct) / K;
    t.transform(ht * G + lt * et, ut * G + dt * et, lt * nt, dt * nt, ht * ft + lt * it + ot, ut * ft + dt * it + mt), y = G, P = et, L = 0, B = nt, j = ft, ct = it;
    const Mt = Z > 0.4 ? 0 : Z >= 0 ? 1 : 2;
    if (p)
      t.drawImage(n.sprites[Mt].c, -xt, -xt, Ft, Ft);
    else {
      const gt = Mt === 0 ? n.near : Mt === 1 ? n.rimG : n.far;
      gt !== k && (t.fillStyle = k = gt), t.fill(e.path);
    }
  }
  t.setTransform(F, x, _, w, C, I);
  const W = 1 / (u * m), V = A * E / u, D = -T * E * W, z = Math.hypot(V, D), U = z > 1e-6 ? g * V / z : 1, Q = z > 1e-6 ? g * D / z : 0, st = 50 * X + Math.hypot(50 * (1 - X), z), tt = (st + 50) / 100, J = (st - 50) / 2, f = 1 + (tt - 1) * U * U, b = (tt - 1) * U * Q, R = 1 + (tt - 1) * Q * Q, Y = Ee(Ee(s.ctm, [u, A * T, 0, m, 0, 0]), [f, b, b, R, J * U - 50 * f - 50 * b, J * Q - 50 * b - 50 * R]);
  if (t.save(), t.setTransform(Y[0], Y[1], Y[2], Y[3], Y[4], Y[5]), t.clip(e.path), t.imageSmoothingEnabled = !0, t.imageSmoothingQuality = "high", t.drawImage(M.c, -xt, -xt, Ft, Ft), t.restore(), s.dev >= 256 && i.rim > 0 && i.highlight > 0) {
    t.save(), t.globalCompositeOperation = "source-atop", t.setTransform(Y[0], Y[1], Y[2], Y[3], Y[4], Y[5]);
    const S = Math.min(0.5, 0.3 * i.rim * Math.min(1.4, i.highlight)), v = t.createLinearGradient(50 + h[0] * 50, 50 + h[1] * 50, 50 - h[0] * 50, 50 - h[1] * 50);
    v.addColorStop(0, `rgba(235,244,255,${S.toFixed(3)})`), v.addColorStop(0.45, `rgba(235,244,255,${(0.35 * S).toFixed(3)})`), v.addColorStop(0.75, "rgba(235,244,255,0)"), t.strokeStyle = v, t.lineJoin = "round", t.lineWidth = 1.3, t.stroke(e.path), t.restore();
  }
  return !0;
}
const kt = 1.5, ce = 0.1, Yt = 17, Aa = 15, wa = 0.9, os = (t, e) => e + (1 - e) * Math.sqrt(Math.max(0, 1 - t * t)), ba = 25, ka = 6.3, xa = { eyes: 1, mouth: -3.5 }, Jt = /* @__PURE__ */ new Map();
function Sa(t, e, s) {
  const a = `${t}|${e}|${s}`;
  let o = Jt.get(a);
  if (!o) {
    const i = Ct(t, -0.3 * e, 0.05 * e), r = Ct(t, -0.12 * e, 0.03 * e), c = [], n = [];
    for (let l = 0; l < Yt; l++) {
      const h = l / (Yt - 1);
      c.push(h > 0.6 ? "" : cs(i, r, h / 0.6)), n.push(h >= 0.5 ? t : cs(i, t, h / 0.5));
    }
    o = {
      base: t,
      far: i,
      near: r,
      light: Ct(t, 0.04 * s),
      dark: Ct(t, -0.3 * e, 0.05 * e),
      capTop: Ct(t, 0.035 * s),
      capBottom: Ct(t, -0.035 * e),
      crispMix: c,
      smoothMix: n,
      grad: null
    }, Jt.size > 200 && Jt.clear(), Jt.set(a, o);
  }
  return o;
}
const rs = (t) => (t.startsWith("hsl(") ? t : Ct(t, 0)).match(/[\d.]+/g).map(Number);
function cs(t, e, s) {
  const a = rs(t), o = rs(e), i = a.map((r, c) => r + (o[c] - r) * s);
  return `hsl(${i[0].toFixed(1)} ${i[1].toFixed(1)}% ${i[2].toFixed(1)}%)`;
}
const Rt = 34, Ta = Math.PI * 1.55, Pa = 57, va = 0.4, hs = -0.28, Nt = /* @__PURE__ */ new Map();
function Fa(t) {
  let e = Nt.get(t);
  return e || (e = { base: Ct(t, 0.1, 0.02), light: Ct(t, 0.3, 0.04), dark: Ct(t, -0.22, 0.08), halo: Ct(t, 0.2) }, Nt.size > 200 && Nt.clear(), Nt.set(t, e)), e;
}
const te = (t, e) => t.replace(")", ` / ${Math.max(0, Math.min(1, e)).toFixed(3)})`);
function ls(t, e, s, a, o, i, r) {
  const c = (r == null ? void 0 : r.strength) ?? 0, n = Math.min(1, e.whirl * c);
  if (n <= 0.01) return;
  const l = (r == null ? void 0 : r.size) ?? 1, h = (r == null ? void 0 : r.width) ?? 1, d = (r == null ? void 0 : r.length) ?? 1, M = (r == null ? void 0 : r.tilt) ?? 1, p = Ta * d, u = Fa(s), A = -e.whirlAngle, m = Pa * l, T = m * va * M * (i ? 1.14 : 0.86), E = Math.atan2(o, a) - hs;
  t.save(), t.rotate(hs), t.translate(0, 5), t.lineCap = "butt";
  const X = (g, F, x, _, w) => {
    t.strokeStyle = _, t.lineWidth = x, t.beginPath(), t.ellipse(0, w, m, T, 0, g, F, !1), t.stroke();
  };
  if (i)
    for (let g = 0; g < Rt; g++) {
      const F = g / Rt, x = A + F * p, _ = x + p / Rt + 0.012;
      if (Math.sin((_ + x) / 2) <= 0) continue;
      const w = Math.pow(1 - F, 1.3);
      X(x, _, (2 + 8 * w) * 1.5 * h, `rgba(0,0,0,${(0.2 * n * w).toFixed(3)})`, 3.5);
    }
  for (let g = 0; g < Rt; g++) {
    const F = g / Rt, x = A + F * p, _ = x + p / Rt + 0.012, w = (_ + x) / 2;
    if (Math.sin(w) > 0 !== i) continue;
    const C = 0.6 + 0.4 * Math.sin(w), I = Math.pow(1 - F, 1.3), k = 1 + 0.18 * Math.sin(F * 9 + 1.2), y = (2 + 8 * I) * C * h * k, P = n * (0.3 + 0.7 * I) * C, L = 0.5 + 0.5 * Math.cos(w - E);
    X(x, _, y * 2.6, te(u.halo, P * 0.2), 0), X(x, _, y * 0.8, te(u.dark, P * 0.45), y * 0.32), X(x, _, y, te(u.base, P * 0.72), 0), X(x, _, y * 0.62, te(u.light, P * 0.78 * (0.4 + 0.6 * L)), -y * 0.16), X(x, _, y * 0.24, `rgba(255,255,255,${(P * 0.9 * (0.15 + 0.85 * L * L)).toFixed(3)})`, -y * 0.3);
  }
  t.restore();
}
function fs(t, e, s, a) {
  const o = e * kt;
  t.clearRect(0, 0, o, o);
  const i = e / 100;
  let r, c;
  if (a.dpr !== void 0)
    r = a.dpr, c = [r, 0, 0, r, 0, 0];
  else if (t.getTransform) {
    const W = t.getTransform();
    c = [W.a, W.b, W.c, W.d, W.e, W.f], r = W.a || 1;
  } else
    r = 1, c = [1, 0, 0, 1, 0, 0];
  const n = a.shadow ?? 0.35, l = a.highlight ?? 1.3, h = Aa * (a.depth ?? 0.65), d = 1 - (1 - wa) * (a.rim ?? 0.5), M = a.spread ?? 1.55, p = (a.light ?? 265) * Math.PI / 180, u = Math.sin(p), A = -Math.cos(p), m = Sa(a.color, n, l), T = Math.cos(s.yaw), E = Math.sin(s.yaw), X = Math.cos(s.pitch), g = Math.sin(s.pitch), F = T * X, x = (W) => Math.abs(W) < 0.22 ? W < 0 ? -0.22 : 0.22 : W, _ = x(T), w = x(X), C = Math.cos(s.roll), I = Math.sin(s.roll), k = s.sx * i, y = s.sy * i, P = 50 * (1 - s.sy) * i, L = Ee(c, [C * k, I * k, -I * y, C * y, o / 2 + s.x * i - I * P, o / 2 + ce * e + s.y * i + C * P]);
  t.save(), t.setTransform(L[0], L[1], L[2], L[3], L[4], L[5]), t.lineCap = "round", t.lineJoin = "round";
  const B = a.shading, j = (W, V, D) => {
    let z = m.near, U = m.base;
    if (B === "crisp") {
      if (!m.grad || m.grad.lx !== u || m.grad.ly !== A) {
        const K = t.createLinearGradient(u * 56, A * 56, -u * 56, -A * 56);
        K.addColorStop(0, m.light), K.addColorStop(0.45, m.near), K.addColorStop(1, m.dark);
        const ht = t.createLinearGradient(u * 46, A * 46, -u * 46, -A * 46);
        ht.addColorStop(0, m.capTop), ht.addColorStop(1, m.capBottom), m.grad = { lx: u, ly: A, lit: K, cap: ht };
      }
      z = m.grad.lit, U = m.grad.cap;
    }
    let Q = !1;
    B === "plastic" && (Q = ga(
      t,
      { ...a, path: W, typeKey: V },
      { cy: _, sy: E, cp: w, sp: g, facing: F, roll: s.roll, halfDepth: D, cap: d, lx: u, ly: A, dev: e * r, ctm: L, still: a.still },
      m,
      null,
      { shadow: n, highlight: l, spread: M, rim: a.rim ?? 0.5 }
    ));
    const st = B === "plastic" && !Q ? "smooth" : B, tt = st === "smooth", J = tt && typeof Path2D == "function" ? new Path2D() : null, f = F >= 0 ? 1 : -1, [b, R, Y, S, v, $] = L;
    let q = null, Z = 1, G = 0, et = 0, nt = 1, ft = 0, it = 0;
    for (let K = 0; K < Yt && !Q; K++) {
      const ut = -1 + 2 * (f > 0 ? K : Yt - 1 - K) / (Yt - 1), lt = os(ut, d), dt = K / (Yt - 1), ot = _ * lt, mt = E * g * lt, Mt = w * lt, gt = ut * E * D - 50 * ot, Lt = -ut * _ * g * D - 50 * mt - 50 * Mt, wt = Z * nt - G * et, _t = nt / wt, Et = -G / wt, Tt = -et / wt, Dt = Z / wt, pe = (et * it - nt * ft) / wt, O = (G * ft - Z * it) / wt;
      t.transform(_t * ot + Tt * mt, Et * ot + Dt * mt, Tt * Mt, Dt * Mt, _t * gt + Tt * Lt + pe, Et * gt + Dt * Lt + O), Z = ot, G = mt, et = 0, nt = Mt, ft = gt, it = Lt;
      let H;
      tt ? H = m.smoothMix[K] : K === Yt - 1 ? H = U : dt > 0.6 ? H = z : H = m.crispMix[K], H !== q && (t.fillStyle = q = H), t.fill(W), J && J.addPath(W, { a: ot, b: mt, c: 0, d: Mt, e: gt, f: Lt });
    }
    if (Q || t.setTransform(b, R, Y, S, v, $), J && st === "smooth") {
      t.save(), t.clip(J);
      const K = Math.min(1, 0.34 * n), ht = t.createRadialGradient(-u * 45, -A * 45, 4 * M, -u * 45, -A * 45, 84 * M);
      ht.addColorStop(0, `rgba(0,0,0,${K})`), ht.addColorStop(0.5, `rgba(0,0,0,${K * 0.35})`), ht.addColorStop(1, "rgba(0,0,0,0)"), t.globalCompositeOperation = "multiply", t.fillStyle = ht, t.fillRect(-120, -120, 240, 240), t.globalCompositeOperation = "source-over";
      const ut = Math.min(1, 0.22 * l), lt = t.createRadialGradient(u * 37, A * 37, 0, u * 37, A * 37, 62 * M);
      lt.addColorStop(0, `rgba(255,255,255,${ut})`), lt.addColorStop(0.6, `rgba(255,255,255,${ut * 0.23})`), lt.addColorStop(1, "rgba(255,255,255,0)"), t.fillStyle = lt, t.fillRect(-120, -120, 240, 240), t.restore();
    }
    return Q;
  };
  ls(t, s, a.color, u, A, !1, a.whirl), a.parts && j(a.parts, `${a.typeKey ?? "custom"}:parts`, h * (a.partsDepth ?? 0.4));
  const ct = j(a.path, a.typeKey ?? "custom", h);
  if (F > -0.2) {
    t.save();
    {
      const W = F >= 0 ? 1 : -1, V = os(W, d), D = _ * V, z = E * g * V, U = w * V, Q = W * E * h - 50 * D, st = -W * _ * g * h - 50 * z - 50 * U, [tt, J, f, b, R, Y] = L;
      t.setTransform(tt * D + f * z, J * D + b * z, f * U, b * U, tt * Q + f * st + R, J * Q + b * st + Y), t.clip(a.path), t.setTransform(tt, J, f, b, R, Y);
    }
    t.translate(a.faceX - 50, a.faceY - 50), t.scale(a.faceScale, a.faceScale), ct && (t.globalAlpha = 0.93), $a(t, s, a), t.restore();
  }
  ls(t, s, a.color, u, A, !0, a.whirl), t.restore();
}
const ee = 30;
function La(t, e, s, a) {
  const o = Math.asin(Math.max(-1, Math.min(1, t / ee))) + s, i = Math.asin(Math.max(-1, Math.min(1, -e / ee))) + a, r = Math.cos(i);
  return {
    x: ee * Math.sin(o) * r,
    y: -ee * Math.sin(i),
    sx: Math.cos(o),
    sy: r,
    z: Math.cos(o) * r
  };
}
const us = 8, se = /* @__PURE__ */ new Map();
function Ea(t, e, s) {
  const a = Math.round(t * 50), o = Math.round(e * 50), i = Math.round(s * 50), r = a + 2e3 * o + 4e6 * i;
  let c = se.get(r);
  if (!c) {
    const n = a / 50, l = o / 50, h = i / 50;
    let d = `M${-n} ${l}`;
    for (let M = 1; M <= us; M++) {
      const p = M / us, u = 1 - p;
      d += ` L${(u * u * -n + p * p * n).toFixed(3)} ${((u * u + p * p) * l + 2 * u * p * h).toFixed(3)}`;
    }
    c = new Path2D(d), se.size > 256 && se.clear(), se.set(r, c);
  }
  return c;
}
const ds = Math.sin(0.684), ps = Math.cos(0.684), ae = /* @__PURE__ */ new Map();
function Ia(t) {
  const e = (w) => Math.round(w * 50) / 50, s = e(t.hw), a = e(t.t0), o = e(t.a), i = e(t.yt), r = e(t.ab), c = e(t.yb), n = `${s},${a},${o},${i},${r},${c}`;
  let l = ae.get(n);
  if (l) return l;
  const h = (w) => w.toFixed(3), d = a * ds, M = a * ps, p = -s + d, u = -M, A = s - d, m = -M, T = -s - d, E = M, X = s + d, g = M, F = 4 / 3 * a * ps, x = 4 / 3 * a * ds, _ = `M${h(p)} ${h(u)}C${h(-s + o * s)} ${h(i - a)} ${h(s - o * s)} ${h(i - a)} ${h(A)} ${h(m)}C${h(A + F)} ${h(m - x)} ${h(X + F)} ${h(g - x)} ${h(X)} ${h(g)}C${h(s - r * s)} ${h(c + a)} ${h(-s + r * s)} ${h(c + a)} ${h(T)} ${h(E)}C${h(T - F)} ${h(E - x)} ${h(p - F)} ${h(u - x)} ${h(p)} ${h(u)}Z`;
  return l = new Path2D(_), ae.size > 256 && ae.clear(), ae.set(n, l), l;
}
function $a(t, e, s) {
  const [a, o, i] = e.w, r = s.ink, c = xa[s.face], n = ba / 2, l = e.lookX, h = e.lookY, { yaw: d, pitch: M } = e, p = (w, C, I, k = 1) => {
    const y = La(w, C, d, M);
    y.z <= 0.02 || k <= 0.01 || (t.save(), t.globalAlpha = k * Math.min(1, y.z * 5), t.translate(y.x, y.y), t.scale(Math.max(0.02, y.sx), Math.max(0.02, y.sy)), I(), t.restore());
  }, u = (w, C) => Math.abs(w) <= C ? 0 : Math.sign(w) * (Math.abs(w) - C) / (1 - C), A = (w) => Math.max(-1, Math.min(1, w)), m = u(A(-e.pitch / 0.26 - e.lookY / 7), 0.34), T = Math.abs(u(A(e.lookX / 4.5), 0.4)), E = Math.max(0.3, 1 + 0.55 * m - 0.1 * T), X = 1 - 0.05 * m + 0.12 * T, g = a + o * (1 - e.laugh), F = o * e.laugh, x = Math.max(0, -e.y) / 26, _ = 0.5 + 0.5 * e.breath;
  for (const w of [-1, 1]) {
    const C = w < 0 ? e.blinkL : e.blinkR, I = Math.max(0, Math.min(1, e.eyeOpen * (1 - C))), k = g * I, y = g * (1 - I), P = F, L = i, B = k * 0.01 + y * 5.4 + P * 6.2 + L * 6, j = k * 1.1 * E + y * 0.6 + P * (2.2 - x * 1.5) + L * (-1.4 + _), ct = k * -3.3 * E + y * 0.6 + P * (-11.4 - 4 * x) + L * (5.4 + 2 * _), W = k * ka * 2 * X + y * 2.8 + P * 4.4 + L * 4, V = l * (k + 0.5 * (y + P)), D = h * (k + 0.5 * y);
    p(w * n + V, c + D, () => {
      t.strokeStyle = r, t.lineWidth = W, t.stroke(Ea(B, j, ct));
    });
  }
  if (s.face === "mouth") {
    const w = l * 0.35, C = (0.6 + 0.4 * a) * (1 + 0.06 * e.breath), I = (0.6 + 0.4 * o) * (1 + 0.25 * Math.max(0, -e.y) / 26), k = 2.7 * i * (1 + 0.25 * e.breath), y = (L, B, j) => a * L + o * B + i * j, P = {
      hw: y(6.5 * C, 9.5 * I, k),
      t0: y(1.9, 0, 0),
      a: y(2 / 3, 2 / 3, 0),
      yt: y(3.53 * C, 1.6 * I, -4 * k / 3),
      ab: y(2 / 3, 0, 0),
      yb: y(3.53 * C, 17.3 * I, 4 * k / 3)
    };
    p(w, y(12.5, 11.6, 15.5), () => {
      t.fillStyle = r, t.fill(Ia(P));
    });
  }
}
const St = { x: NaN, y: NaN }, Wt = /* @__PURE__ */ new Set();
let At = 0, he = 0;
function Is(t) {
  At = 0;
  const e = he ? Math.min(0.1, (t - he) / 1e3) : 0;
  he = t, Wt.forEach((s) => s(e)), Wt.size && (At = requestAnimationFrame(Is));
}
function $s() {
  At || typeof document > "u" || document.hidden || (he = 0, At = requestAnimationFrame(Is));
}
let ms = !1;
function Ya() {
  ms || typeof document > "u" || (ms = !0, document.addEventListener("pointermove", (t) => {
    St.x = t.clientX, St.y = t.clientY;
  }, { passive: !0 }), document.addEventListener("pointerleave", () => {
    St.x = NaN, St.y = NaN;
  }), window.addEventListener("blur", () => {
    St.x = NaN, St.y = NaN;
  }), document.addEventListener("visibilitychange", () => {
    document.hidden ? (At && cancelAnimationFrame(At), At = 0) : Wt.size && $s();
  }));
}
function Xa(t) {
  return Ya(), Wt.add(t), $s(), () => {
    Wt.delete(t), !Wt.size && At && (cancelAnimationFrame(At), At = 0);
  };
}
function Ms(t) {
  let e = 2166136261;
  for (let s = 0; s < t.length; s++) e = Math.imul(e ^ t.charCodeAt(s), 16777619);
  return (e >>> 0) % 1e3 / 1e3;
}
const yt = (t, e, s) => Math.min(s, Math.max(e, Number.isFinite(t) ? t : 1)), Cs = /* @__PURE__ */ new Map();
function ys(t) {
  let e = Cs.get(t);
  return e || (e = new Path2D(t), Cs.set(t, e)), e;
}
const ve = () => typeof matchMedia == "function" && matchMedia("(prefers-reduced-motion: reduce)").matches, Oa = qs(function({
  type: e = "clover",
  path: s,
  face: a,
  state: o = "default",
  size: i = 64,
  color: r,
  ink: c,
  brightness: n = 1,
  saturation: l = 1.5,
  speed: h = 1,
  paused: d = !1,
  seed: M,
  shading: p = "plastic",
  shadow: u = 0.35,
  highlight: A = 1.3,
  depth: m = 0.65,
  light: T = 265,
  rim: E = 0.5,
  spread: X = 1.55,
  interactive: g = !0,
  turn: F = 1,
  theme: x = "auto",
  whirl: _ = 0,
  whirlSize: w = 1,
  whirlWidth: C = 1,
  whirlLength: I = 1,
  whirlTilt: k = 1,
  jumpHeight: y = 26,
  jumpTime: P = 0.68,
  jumpStretch: L = 1,
  jumpSpin: B = 1,
  jumpLean: j = 6,
  jumpEvery: ct = 8,
  jumpLand: W = 0,
  jumpSquash: V = 1.15,
  jumpSquashTime: D = 0.37,
  jumpSquashEase: z = "pulse",
  jumpGroundTime: U = 0.11,
  jumpGroundEase: Q = "pulse",
  jumpRiseTime: st = 0.33,
  jumpRiseEase: tt = "pulse",
  jumpClickSquashTime: J = 0.24,
  className: f,
  style: b,
  "aria-label": R,
  ...Y
}, S) {
  const v = Xt(null);
  Rs(S, () => v.current);
  const $ = Ws(), q = le[e] ?? le.clover, Z = a ?? q.face, G = r ?? q.color, et = n === 1 && l === 1 ? G : Ct(G, (Math.min(2, Math.max(0, n)) - 1) * 0.35, (Math.min(2, Math.max(0, l)) - 1) * 0.5), nt = c ?? Vs(et), ft = Math.min(1, Math.max(0, M ?? Ms($))), it = o in qe ? o : "default", K = d || !(h > 0), ht = p === !0 ? "crisp" : p === !1 ? "flat" : p, ut = typeof s == "string" && s.trim() ? s.trim() : null, lt = ut ? `path:${Ms(ut)}:${ut.length}` : e, dt = Xt(null), ot = Xt(null), mt = Xt(0), Mt = Xt(h);
  Mt.current = h;
  const gt = Xt(g);
  gt.current = g, ot.current = {
    path: typeof Path2D > "u" ? null : ys(ut ?? Re[e] ?? Re.clover),
    face: Z,
    faceX: q.faceX,
    faceY: q.faceY,
    faceScale: q.faceScale,
    color: et,
    ink: nt,
    shading: ht,
    shadow: yt(u, 0, 2),
    highlight: yt(A, 0, 2),
    depth: yt(m, 0.2, 2),
    light: T,
    rim: yt(E, 0, 2),
    spread: yt(X, 0.4, 2.5),
    typeKey: lt,
    still: K || ve(),
    whirl: { strength: yt(_, 0, 2), size: yt(w, 0.6, 1.6), width: yt(C, 0.4, 2), length: yt(I, 0.4, 1.6), tilt: yt(k, 0.5, 1.8) },
    parts: typeof Path2D < "u" && !ut && We[e] ? ys(We[e]) : void 0
  };
  const Lt = (O) => {
    if (x !== "auto") return x;
    const H = O == null ? void 0 : O.closest("[data-theme], .dark, .light");
    if (H) {
      const N = H.getAttribute("data-theme");
      if (N === "dark" || N === "light") return N;
      if (H.classList.contains("dark")) return "dark";
      if (H.classList.contains("light")) return "light";
    }
    return typeof matchMedia == "function" && matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }, wt = () => {
    const O = v.current, H = ot.current;
    if (!O || !H || !H.path) return;
    const N = O.clientWidth / kt || mt.current || (typeof i == "number" ? i : 64);
    if (!N) return;
    const pt = Math.min(2, typeof devicePixelRatio == "number" && devicePixelRatio || 1), bt = Math.round(N * kt * pt);
    (O.width !== bt || O.height !== bt) && (O.width = bt, O.height = bt), mt.current = N;
    const Pt = O.getContext("2d");
    if (!Pt) return;
    Pt.setTransform(pt, 0, 0, pt, 0, 0), H.dpr = pt;
    const Gt = dt.current ? dt.current.pose : Be(it);
    fs(Pt, N, Gt, H);
  };
  _s(() => {
    if (dt.current ? dt.current.setState(it) : dt.current = new ta(ft, it), dt.current.setTurn(yt(F, 0, 2)), dt.current.setJump({ height: y, time: Math.max(0.2, P), stretch: L, spin: Math.max(0, Math.round(B)), lean: j, every: ct, land: W, squash: V, squashTime: Math.max(0.05, D), squashEase: z, groundTime: Math.max(0, U), groundEase: Q, riseTime: Math.max(0.05, st), riseEase: tt, clickSquashTime: Math.max(0.05, J) }), ve()) {
      const O = v.current;
      if (O && ot.current && ot.current.path) {
        const H = O.clientWidth / kt || mt.current || (typeof i == "number" ? i : 64);
        if (!H) return;
        mt.current = H;
        const N = Math.min(2, typeof devicePixelRatio == "number" && devicePixelRatio || 1);
        O.width = O.height = Math.round(H * kt * N);
        const pt = O.getContext("2d");
        pt && (ot.current.theme = Lt(O), ot.current.dpr = N, pt.setTransform(N, 0, 0, N, 0, 0), fs(pt, H, Be(it), ot.current));
      }
      return;
    }
    ot.current && v.current && (ot.current.theme = Lt(v.current)), wt();
  }), Xe(() => {
    var bt;
    if (ht !== "plastic" || !((bt = ot.current) != null && bt.path)) return;
    const O = ot.current.path, H = (typeof i == "number" ? i : 64) * Math.min(2, typeof devicePixelRatio == "number" && devicePixelRatio || 1), pt = (typeof requestIdleCallback == "function" ? requestIdleCallback : (Pt) => setTimeout(Pt, 1))(() => ua(lt, O, H, m));
    return () => {
      typeof cancelIdleCallback == "function" ? cancelIdleCallback(pt) : clearTimeout(pt);
    };
  }, [ht, lt, i, m]), Xe(() => {
    if (K || ve()) return;
    const O = v.current;
    if (!O) return;
    let H = !0, N = null;
    const pt = 3, bt = (me) => {
      const It = dt.current;
      if (It) {
        if (gt.current && !Number.isNaN(St.x)) {
          const Zt = O.getBoundingClientRect(), Me = Zt.width / kt || 1, $e = (St.x - (Zt.left + Zt.width / 2)) / Me, Ye = (St.y - (Zt.top + Zt.height / 2 + ce * Me)) / Me, Ot = Math.hypot($e, Ye), Ys = Ot < 1 ? 1 : Ot > pt ? 0 : 1 - (Ot - 1) / (pt - 1);
          It.setPointer($e / Math.max(1, Ot), Ye / Math.max(1, Ot), Ys);
        } else It.setPointer(0, 0, 0);
        It.update(me * Mt.current), wt();
      }
    }, Pt = () => {
      N || (N = Xa(bt));
    }, Gt = () => {
      N && N(), N = null;
    };
    let Bt = null;
    return typeof IntersectionObserver == "function" ? (Bt = new IntersectionObserver((me) => {
      var It;
      H = ((It = me[0]) == null ? void 0 : It.isIntersecting) ?? !0, H ? Pt() : Gt();
    }), Bt.observe(O)) : Pt(), () => {
      Gt(), Bt && Bt.disconnect();
    };
  }, [K]);
  const _t = typeof i == "number" ? `${i * kt}px` : `calc(${i} * ${kt})`, Et = (O) => typeof i == "number" ? `${-i * O}px` : `calc(${i} * ${-O})`, Tt = (kt - 1) / 2, Dt = {
    display: "inline-block",
    verticalAlign: "middle",
    width: _t,
    height: _t,
    marginLeft: Et(Tt),
    marginRight: Et(Tt),
    marginTop: Et(Tt + ce),
    marginBottom: Et(Tt - ce),
    flex: "none",
    ...b
  }, pe = (O) => {
    var H, N;
    g && !K && ((H = dt.current) == null || H.poke()), (N = Y.onClick) == null || N.call(Y, O);
  };
  return /* @__PURE__ */ Xs(
    "canvas",
    {
      ref: v,
      className: f ? `ba ${f}` : "ba",
      "data-bot-avatar": e,
      "data-face": Z,
      "data-state": it,
      role: "img",
      "aria-label": R ?? `${q.label} bot, ${qe[it]}`,
      style: Dt,
      ...Y,
      onClick: pe
    }
  );
});
export {
  Za as BOT_AVATAR_MATCAP_SIZE,
  kt as BOT_AVATAR_OVERSCAN,
  xt as BOT_AVATAR_PAD,
  ce as BOT_AVATAR_RISE,
  Ft as BOT_AVATAR_SPAN,
  Oa as BotAvatar,
  ta as BotAvatarSim,
  Vs as autoInk,
  ca as bakeBotAvatarForm,
  Ma as botAvatarCapFrame,
  Wa as botAvatarFaces,
  De as botAvatarJumpDefaults,
  Da as botAvatarPalette,
  We as botAvatarParts,
  le as botAvatarPresets,
  Re as botAvatarShapes,
  _a as botAvatarStates,
  Ps as botAvatarTier,
  Ds as botAvatarTypes,
  pa as buildBotAvatarMatcap,
  Oa as default,
  fs as drawBotAvatarFrame,
  Zs as luminance,
  Ie as parseColor,
  Be as restPose,
  Ct as shade,
  ma as shadeBotAvatarTexels,
  ua as warmBotAvatarPlastic
};
