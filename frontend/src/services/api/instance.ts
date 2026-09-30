import { config } from "../../app/config";
import { getAccessToken } from "../supabase/auth";
import { ApiClient } from "./client";

/** The app-wide client: attaches the Supabase access token when signed in, sends guests without one. */
export const apiClient = new ApiClient(config.apiUrl, getAccessToken);
