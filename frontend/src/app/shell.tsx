import { useEffect, useState, type CSSProperties } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import {
  History,
  House,
  Languages,
  LogIn,
  LogOut,
  Hand,
  Settings,
} from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarRail,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { ModeToggle } from "@/components/mode-toggle";
import { SignSageLogo } from "@/components/logo";
import { apiClient } from "@/services/api/instance";
import { getAccessToken, onAuthChange, signOut } from "@/services/supabase/auth";
import { cn } from "@/lib/utils";

const ACTIVE =
  "bg-green-500/10 font-medium text-foreground [&>svg]:text-green-600 dark:[&>svg]:text-green-400";

const INTERPRET_NAV = [
  { to: "/app/interpreter", label: "Live Interpreter", icon: Languages },
  { to: "/app/generator", label: "Sign Generator", icon: Hand },
];

const LIBRARY_NAV = [
  { to: "/app/translation-log", label: "Translation Log", icon: History },
];

const TITLES: Record<string, string> = {
  "/app": "Live Interpreter",
  "/app/interpreter": "Live Interpreter",
  "/app/generator": "Sign Generator",
  "/app/translation-log": "Translation Log",
  "/app/settings": "Settings",
};

function UserArea() {
  const [email, setEmail] = useState<string | null>(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    const live = { current: true };
const resolve = async () => {
      const token = await getAccessToken();
      if (!live.current) return;
      if (!token) {
        setEmail(null);
        setChecking(false);
        return;
      }
      try {
        const me = await apiClient.me();
        if (live.current) setEmail(me.email ?? "Account");
      } catch {
        if (live.current) setEmail("Account");
      }
      setChecking(false);
    };
    void resolve();
    return onAuthChange((signedIn) => {
      if (!signedIn) {
        setEmail(null);
        setChecking(false);
      } else {
        setChecking(true);
        void resolve();
      }
    });
  }, []);

  if (checking) {
    return <Skeleton className="h-9 w-full" aria-label="Checking sign-in status" />;
  }

  if (!email) {
    return (
      <Button variant="ghost" size="sm" className="w-full justify-start" render={<Link to="/login" />}>
        <LogIn aria-hidden="true" />
        <span className="group-data-[collapsible=icon]:hidden">Sign in</span>
      </Button>
    );
  }

  const initial = email.charAt(0).toUpperCase();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <button
            type="button"
            aria-label="Account menu"
            className="flex w-full items-center gap-2 rounded-lg px-1 py-1 text-left transition-colors hover:bg-muted"
          />
        }
      >
        <Avatar className="h-8 w-8">
          <AvatarFallback>{initial}</AvatarFallback>
        </Avatar>
        <span className="min-w-0 flex-1 truncate text-sm font-medium group-data-[collapsible=icon]:hidden">{email}</span>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel className="truncate font-normal text-foreground/60">
          {email}
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onSelect={() => {
            void signOut().then(() => {
              setEmail(null);
            });
          }}
        >
          <LogOut aria-hidden="true" />
          Sign out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function NavItems({
  items,
}: {
  items: { to: string; label: string; icon: typeof Languages }[];
}) {
  return (
    <SidebarMenu>
      {items.map((item) => (
        <SidebarMenuItem key={item.to}>
          <SidebarMenuButton
            tooltip={item.label}
            render={
              <NavLink
                to={item.to}
                className={({ isActive }) => cn(isActive && ACTIVE)}
              />
            }
          >
            <item.icon aria-hidden="true" />
            <span>{item.label}</span>
          </SidebarMenuButton>
        </SidebarMenuItem>
      ))}
    </SidebarMenu>
  );
}

function AppSidebar() {
  return (
    <Sidebar variant="inset" collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            {/* One toggle only: here. In icon-collapsed mode the logo row
                hides and this trigger becomes the centered expand control
                (the rail stays available too); keyboard reachable. */}
            <div className="flex w-full items-center gap-1 group-data-[collapsible=icon]:justify-center">
              <SidebarMenuButton
                size="lg"
                className="flex-1 group-data-[collapsible=icon]:hidden"
                render={<Link to="/" />}
              >
                <SignSageLogo className="h-8 w-8" />
                <span className="font-semibold tracking-tight">
                  SignSage
                </span>
              </SidebarMenuButton>
              <SidebarTrigger
                aria-label="Toggle navigation"
                className="group-data-[collapsible=icon]:size-8"
              />
            </div>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Interpret</SidebarGroupLabel>
          <SidebarGroupContent>
            <NavItems items={INTERPRET_NAV} />
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup>
          <SidebarGroupLabel>Library</SidebarGroupLabel>
          <SidebarGroupContent>
            <NavItems items={LIBRARY_NAV} />
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              tooltip="Settings"
              render={
                <NavLink to="/app/settings" className={({ isActive }) => cn(isActive && ACTIVE)} />
              }
            >
              <Settings aria-hidden="true" />
              <span>Settings</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
        <Separator />
        <div className="flex items-center justify-between gap-2 px-1 py-1 group-data-[collapsible=icon]:hidden">
          <span className="text-sm text-foreground/60">Appearance</span>
          <ModeToggle />
        </div>
        <Separator />
        <UserArea />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}

export function AppShell() {
  const { pathname } = useLocation();
  const title = TITLES[pathname] ?? "Live Interpreter";

  return (
    <SidebarProvider style={{ "--sidebar-width": "15rem" } as CSSProperties}>
      <AppSidebar />
      <SidebarInset>
        <header className="sticky top-0 z-10 flex h-14 items-center gap-2 border-b border-foreground/10 bg-background/80 px-4 backdrop-blur-md">
          {/* Offcanvas trigger: only below md (sheet mode). Desktop uses the sidebar's own toggle. */}
          <SidebarTrigger aria-label="Toggle navigation" className="md:hidden" />
          <Separator orientation="vertical" className="h-5 md:hidden" />
          <Breadcrumb className="min-w-0">
            <BreadcrumbList className="flex-nowrap">
              <BreadcrumbItem className="shrink-0">
                <Link
                  to="/"
                  className="flex items-center gap-1.5 text-foreground/60 transition-colors hover:text-foreground"
                >
                  <House aria-hidden="true" className="h-3.5 w-3.5" />
                  SignSage
                </Link>
              </BreadcrumbItem>
              <BreadcrumbSeparator />
              <BreadcrumbItem className="min-w-0">
                <BreadcrumbPage className="truncate">{title}</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </header>
        <main className="min-h-[calc(100svh-3.5rem)] bg-background text-foreground">
          <div className="mx-auto w-full max-w-6xl px-4 py-8 md:px-8">
            <Outlet />
          </div>
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
}
