import { createBrowserRouter, Navigate } from "react-router-dom";
import { AppShell } from "./shell";
import LandingPage from "@/features/landing/LandingPage";
import AboutPage from "@/features/about/AboutPage";
import { LoginPage, SignupPage, StartPage } from "@/features/auth/AuthPages";
import HowItWorksPage from "@/features/how-it-works/HowItWorksPage";
import CapabilitiesPage from "@/features/capabilities/CapabilitiesPage";
import PrivacyPage from "@/features/privacy/PrivacyPage";
import TranslatePage from "@/features/translate/TranslatePage";
import TextToSignPage from "@/features/text-to-sign/TextToSignPage";
import HistoryPage from "@/features/history/HistoryPage";
import SavedSignsPage from "@/features/saved/SavedSignsPage";
import SettingsPage from "@/features/settings/SettingsPage";
import HelpPage from "@/features/help/HelpPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <LandingPage />,
  },
  {
    path: "/about",
    element: <AboutPage />,
  },
  {
    path: "/how-it-works",
    element: <HowItWorksPage />,
  },
  {
    path: "/help",
    element: <HelpPage />,
  },
  {
    path: "/capabilities",
    element: <CapabilitiesPage />,
  },
  {
    path: "/privacy",
    element: <PrivacyPage />,
  },
  {
    path: "/login",
    element: <LoginPage />,
  },
  {
    path: "/signup",
    element: <SignupPage />,
  },
  {
    path: "/start",
    element: <StartPage />,
  },
  {
    path: "/app",
    element: <AppShell />,
    children: [      {
        index: true,
        element: <Navigate to="interpreter" replace />,
      },
      {
        path: "interpreter",
        element: <TranslatePage />,
      },
      {
        path: "generator",
        element: <TextToSignPage />,
      },
      {
        path: "translation-log",
        element: <HistoryPage />,
      },
      {
        path: "custom-glossary",
        element: <SavedSignsPage />,
      },
      {
        path: "settings",
        element: <SettingsPage />,
      },
    ],
  },
  {
    path: "*",
    element: <Navigate to="/" replace />,
  },
]);
