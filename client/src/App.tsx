import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createSimulation,
  deleteSimulation,
  devAuthBypass,
  fetchAuthConfig,
  fetchMe,
  fetchVisitors,
  googleLogin,
  fetchOverview,
  fetchSimulations,
  fetchStrategies,
  fetchWatchlist,
  fetchSparkline,
  fetchChart,
  searchSymbols,
  login,
  predictStrategy,
  signup,
  trainStrategy,
  updateSimulation,
  askChat,
  type AuthResponse,
  type DevAuthBypassPayload,
} from "./api";
import { Dashboard } from "./components/Dashboard";
import { HomeOverview } from "./components/HomeOverview";
import { ChatbotPanel } from "./components/ChatbotPanel";
import { StrategyCatalog } from "./components/StrategyCatalog";
import { LiveMarketPage } from "./components/LiveMarketPage";
import { LoginForm } from "./components/LoginForm";
import { SignupForm } from "./components/SignupForm";
import { GoogleLoginPage } from "./components/GoogleLoginPage";
import { googleSignOut } from "./components/GoogleSignIn";
import { VisitorsPage } from "./components/VisitorsPage";
import {
  ChatIcon,
  HomeIcon,
  LiveIcon,
  LogoMark,
  LogoutIcon,
  SimulationsIcon,
  StrategyIcon,
  VisitorsIcon,
} from "./components/Icons";
import type {
  AuthConfig,
  ChatAction,
  ChatMessage,
  MarketQuote,
  OverviewResponse,
  PredictionResult,
  Simulation,
  SimulationInput,
  SimulationUpdate,
  StrategyDefinition,
  TrainingPayload,
  TrainingResult,
  SparklineSeries,
  User,
} from "./types";

const WATCHLIST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"];

const TRUTHY_ENV_FLAGS = new Set(["1", "true", "yes", "on"]);

type AuthView = "login" | "signup" | "dashboard";
type Page = "home" | "simulations" | "chat" | "strategies" | "live" | "visitors";

interface SessionState {
  token: string;
  user: User;
}

function isEnvFlagEnabled(value: string | undefined) {
  if (!value) {
    return false;
  }
  return TRUTHY_ENV_FLAGS.has(value.trim().toLowerCase());
}

const LOGIN_BYPASS_ENABLED = isEnvFlagEnabled(import.meta.env.VITE_ENABLE_LOGIN_BYPASS);
const LOGIN_BYPASS_EMAIL = import.meta.env.VITE_LOGIN_BYPASS_EMAIL;
const LOGIN_BYPASS_NAME = import.meta.env.VITE_LOGIN_BYPASS_NAME;
const LOGIN_BYPASS_PATH = "/dev/auth/bypass";

const NAV_ITEMS: Array<{ page: Page; label: string; icon: typeof HomeIcon; adminOnly?: boolean }> = [
  { page: "home", label: "Overview", icon: HomeIcon },
  { page: "simulations", label: "Simulations", icon: SimulationsIcon },
  { page: "chat", label: "Trading copilot", icon: ChatIcon },
  { page: "strategies", label: "Strategies", icon: StrategyIcon },
  { page: "live", label: "Live markets", icon: LiveIcon },
  { page: "visitors", label: "Visitors", icon: VisitorsIcon, adminOnly: true },
];

function extractHistory(messages: ChatMessage[]): Array<{ role: "user" | "assistant"; content: string }> {
  return messages
    .filter((message) => message.role === "user" || message.role === "assistant")
    .slice(-8)
    .map((message) => ({ role: message.role, content: message.content }));
}

export default function App() {
  const [view, setView] = useState<AuthView>("login");
  const [page, setPage] = useState<Page>("home");
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [watchlist, setWatchlist] = useState<MarketQuote[]>([]);
  const [simulations, setSimulations] = useState<Simulation[]>([]);
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [strategies, setStrategies] = useState<StrategyDefinition[]>([]);
  const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [sparklines, setSparklines] = useState<SparklineSeries[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [labLoading, setLabLoading] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [authConfig, setAuthConfig] = useState<AuthConfig | null>(null);

  useEffect(() => {
    fetchAuthConfig()
      .then(setAuthConfig)
      // Older or unreachable backends: fall back to email/password and the dev bypass.
      .catch(() => setAuthConfig({ googleClientId: null, passwordLogin: true, devBypass: true }));
  }, []);

  const [bypassRequestedFromPath] = useState(() => {
    if (typeof window === "undefined") {
      return false;
    }
    return window.location.pathname === LOGIN_BYPASS_PATH;
  });
  const loginBypassEnabled = LOGIN_BYPASS_ENABLED || bypassRequestedFromPath;
  const [initializingBypass, setInitializingBypass] = useState(loginBypassEnabled);
  const bypassInProgressRef = useRef(false);
  const [bypassFailed, setBypassFailed] = useState(false);
  const [bypassDismissed, setBypassDismissed] = useState(false);

  const bypassPayload = useMemo<DevAuthBypassPayload | undefined>(() => {
    const email = LOGIN_BYPASS_EMAIL?.trim();
    const name = LOGIN_BYPASS_NAME?.trim();
    if (!email && !name) {
      return undefined;
    }
    return {
      email: email || undefined,
      name: name || undefined,
    };
  }, []);

  const shouldAttemptBypass =
    loginBypassEnabled && authConfig?.devBypass === true && !bypassFailed && !bypassDismissed && !token && !user;

  useEffect(() => {
    const stored = window.localStorage.getItem("algo-trade-session");
    if (stored) {
      try {
        const parsed = JSON.parse(stored) as SessionState;
        if (parsed.token && parsed.user) {
          setToken(parsed.token);
          setUser(parsed.user);
          setView("dashboard");
        }
      } catch (storageError) {
        console.warn("Failed to parse stored session", storageError);
        window.localStorage.removeItem("algo-trade-session");
      }
    }
  }, []);

  useEffect(() => {
    if (token && user) {
      const payload: SessionState = { token, user };
      window.localStorage.setItem("algo-trade-session", JSON.stringify(payload));
    } else {
      window.localStorage.removeItem("algo-trade-session");
    }
  }, [token, user]);

  const loadOverview = useCallback(async () => {
    if (!token) {
      return;
    }
    try {
      const response = await fetchOverview(token);
      setOverview(response);
      const sparklineSeries = await fetchSparkline(token, WATCHLIST_SYMBOLS);
      setSparklines(sparklineSeries);
    } catch (loadError) {
      if (handleAuthFailure(loadError)) {
        return;
      }
      console.error(loadError);
      setError(loadError instanceof Error ? loadError.message : "Unable to load overview data.");
    }
  }, [token]);

  const refreshAll = useCallback(async () => {
    if (!token) {
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const [quotes, storedSimulations, overviewResponse] = await Promise.all([
        fetchWatchlist(WATCHLIST_SYMBOLS),
        fetchSimulations(token),
        fetchOverview(token),
      ]);
      const sparklineSeries = await fetchSparkline(token, WATCHLIST_SYMBOLS);
      setWatchlist(quotes);
      setSimulations(storedSimulations);
      setOverview(overviewResponse);
      setSparklines(sparklineSeries);
    } catch (loadError) {
      if (handleAuthFailure(loadError)) {
        return;
      }
      console.error(loadError);
      setError(loadError instanceof Error ? loadError.message : "Unable to load dashboard data.");
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    if (view === "dashboard" && token) {
      void refreshAll();
    }
  }, [view, token, refreshAll]);

  useEffect(() => {
    if (!token) {
      return;
    }
    void (async () => {
      try {
        const defs = await fetchStrategies();
        setStrategies(defs);
      } catch (catalogError) {
        console.error(catalogError);
      }
    })();
  }, [token]);

  const handleAuthSuccess = useCallback((response: AuthResponse) => {
    setToken(response.token);
    setUser(response.user);
    setView("dashboard");
    setPage("home");
  }, []);

  const handleLogin = useCallback(
    async (email: string, password: string) => {
      setLoading(true);
      setError(null);
      try {
        const response = await login({ email, password });
        handleAuthSuccess(response);
      } catch (authError) {
        console.error(authError);
        setError(authError instanceof Error ? authError.message : "Unable to sign in.");
      } finally {
        setLoading(false);
      }
    },
    [handleAuthSuccess],
  );

  const handleGoogleCredential = useCallback(
    async (credential: string) => {
      setLoading(true);
      setError(null);
      try {
        handleAuthSuccess(await googleLogin(credential));
      } catch (authError) {
        console.error(authError);
        setError(authError instanceof Error ? authError.message : "Google sign-in failed.");
      } finally {
        setLoading(false);
      }
    },
    [handleAuthSuccess],
  );

  const handleSignup = useCallback(
    async (name: string, email: string, password: string) => {
      setLoading(true);
      setError(null);
      try {
        const response = await signup({ name, email, password });
        handleAuthSuccess(response);
      } catch (authError) {
        console.error(authError);
        setError(authError instanceof Error ? authError.message : "Unable to create account.");
      } finally {
        setLoading(false);
      }
    },
    [handleAuthSuccess],
  );

  useEffect(() => {
    if (!loginBypassEnabled) {
      setInitializingBypass(false);
      return;
    }

    if (bypassInProgressRef.current || !authConfig) {
      return; // wait for the auth config: a Google-only deployment has no bypass
    }

    if (!shouldAttemptBypass) {
      setInitializingBypass(false);
      return;
    }

    // A ref (not state) guards the request so re-renders and StrictMode's double effect
    // don't cancel the in-flight login and leave the app stuck on "Signing you in...".
    bypassInProgressRef.current = true;
    setInitializingBypass(true);
    setError(null);
    setLoading(true);

    const run = async () => {
      try {
        const response = await devAuthBypass(bypassPayload);
        setBypassFailed(false);
        handleAuthSuccess(response);
      } catch (bypassError) {
        console.error(bypassError);
        setBypassFailed(true);
        setError(
          bypassError instanceof Error
            ? `Login bypass failed: ${bypassError.message}`
            : "Login bypass is unavailable. Please sign in manually.",
        );
      } finally {
        bypassInProgressRef.current = false;
        setLoading(false);
        setInitializingBypass(false);
      }
    };

    void run();
  }, [loginBypassEnabled, authConfig, shouldAttemptBypass, bypassPayload, handleAuthSuccess]);

  const handleCreateSimulation = useCallback(
    async (payload: SimulationInput) => {
      if (!token) return;
      setLoading(true);
      setError(null);
      try {
        const created = await createSimulation(token, payload);
        setSimulations((previous) => [created, ...previous]);
        void loadOverview();
      } catch (simulationError) {
        if (handleAuthFailure(simulationError)) {
          return;
        }
        console.error(simulationError);
        setError(
          simulationError instanceof Error ? simulationError.message : "Unable to create simulation.",
        );
      } finally {
        setLoading(false);
      }
    },
    [token, loadOverview],
  );

  const handleUpdateSimulation = useCallback(
    async (id: string, update: SimulationUpdate) => {
      if (!token) return;
      setLoading(true);
      setError(null);
      try {
        const updated = await updateSimulation(token, id, update);
        setSimulations((previous) =>
          previous.map((simulation) => (simulation.id === updated.id ? updated : simulation)),
        );
        void loadOverview();
      } catch (updateError) {
        if (handleAuthFailure(updateError)) {
          return;
        }
        console.error(updateError);
        setError(updateError instanceof Error ? updateError.message : "Unable to update simulation.");
      } finally {
        setLoading(false);
      }
    },
    [token, loadOverview],
  );

  const handleDeleteSimulation = useCallback(
    async (id: string) => {
      if (!token) return;
      setLoading(true);
      setError(null);
      try {
        await deleteSimulation(token, id);
        setSimulations((previous) => previous.filter((simulation) => simulation.id !== id));
        void loadOverview();
      } catch (deleteError) {
        if (handleAuthFailure(deleteError)) {
          return;
        }
        console.error(deleteError);
        setError(deleteError instanceof Error ? deleteError.message : "Unable to delete simulation.");
      } finally {
        setLoading(false);
      }
    },
    [token, loadOverview],
  );

  const handleTrainStrategy = useCallback(
    async (payload: TrainingPayload) => {
      if (!token) return;
      setLabLoading(true);
      setError(null);
      try {
        const result = await trainStrategy(token, payload);
        setTrainingResult(result);
        void loadOverview();
      } catch (trainError) {
        if (handleAuthFailure(trainError)) {
          return;
        }
        console.error(trainError);
        setError(trainError instanceof Error ? trainError.message : "Unable to train strategy.");
      } finally {
        setLabLoading(false);
      }
    },
    [token, loadOverview],
  );

  const handlePredictStrategy = useCallback(
    async (symbol: string) => {
      if (!token) return;
      setLabLoading(true);
      setError(null);
      try {
        const trained =
          trainingResult && trainingResult.symbol === symbol.toUpperCase()
            ? { strategyId: trainingResult.strategyId, parameters: trainingResult.parameters }
            : undefined;
        const result = await predictStrategy(token, symbol, trained);
        setPredictionResult(result);
      } catch (predictError) {
        if (handleAuthFailure(predictError)) {
          return;
        }
        console.error(predictError);
        setError(predictError instanceof Error ? predictError.message : "Unable to generate prediction.");
      } finally {
        setLabLoading(false);
      }
    },
    [token, trainingResult],
  );

  const resetSession = useCallback(() => {
    setToken(null);
    setUser(null);
    setSimulations([]);
    setWatchlist([]);
    setOverview(null);
    setTrainingResult(null);
    setPredictionResult(null);
    setSparklines([]);
    setChatMessages([]);
    setView("login");
  }, []);

  // An explicit logout stops the dev bypass from signing the user straight back in.
  const handleLogout = useCallback(() => {
    setBypassDismissed(true);
    googleSignOut();
    resetSession();
  }, [resetSession]);

  const handleAuthFailure = useCallback((issue: unknown) => {
    if (issue && typeof issue === "object" && "status" in (issue as { status?: number })) {
      const status = (issue as { status?: number }).status;
      if (status === 401) {
        if (typeof window !== "undefined") {
          window.localStorage.removeItem("algo-trade-session");
        }
        resetSession();
        setError("Session expired. Please sign in again.");
        return true;
      }
    }
    return false;
  }, [resetSession]);

  // Stored sessions predate role changes, so ask the server who we are (isAdmin, profile photo).
  useEffect(() => {
    if (!token) return;
    fetchMe(token)
      .then((me) => setUser((current) => (current && current.id === me.id ? { ...current, ...me } : current)))
      .catch((meError) => handleAuthFailure(meError));
  }, [token, handleAuthFailure]);

  // Each page starts at the top instead of inheriting the previous page's scroll position.
  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [page]);

  const loadVisitors = useCallback(() => fetchVisitors(token ?? ""), [token]);

  const handleSearchSymbols = useCallback(
    async (query: string) => {
      if (!token) {
        return [];
      }
      return searchSymbols(token, query);
    },
    [token],
  );

  const handleFetchChart = useCallback(
    async (symbol: string, options: { range: string; interval: string }) => {
      if (!token) {
        throw new Error('You need to sign in to view live charts.');
      }
      return fetchChart(token, symbol, options);
    },
    [token],
  );
  const handleChatSend = useCallback(
    async (message: string) => {
      if (!token) return;
      const timestamp = new Date().toISOString();
      const userMessage: ChatMessage = { role: "user", content: message, timestamp };
      const historyForRequest = [...chatMessages, userMessage];
      setChatMessages((previous) => [...previous, userMessage]);
      setChatLoading(true);
      try {
        const response = await askChat(token, {
          message,
          history: extractHistory(historyForRequest),
        });
        const assistantMessage: ChatMessage = {
          role: "assistant",
          content: response.reply,
          timestamp: new Date().toISOString(),
          citations: response.citations,
          actions: response.actions,
        };
        setChatMessages((previous) => [...previous, assistantMessage]);
        if (response.actions?.length) {
          response.actions.forEach((action: ChatAction) => {
            if (action.type === "simulation" && action.data) {
              const simulation = action.data as unknown as Simulation;
              setSimulations((previous) => {
                const exists = previous.find((item) => item.id === simulation.id);
                if (exists) {
                  return previous;
                }
                return [simulation, ...previous];
              });
            }
          });
          void loadOverview();
        }
      } catch (chatError) {
        if (handleAuthFailure(chatError)) {
          return;
        }
        console.error(chatError);
        setChatMessages((previous) => [
          ...previous,
          {
            role: "assistant",
            content:
              chatError instanceof Error
                ? `I ran into an issue reaching the assistant: ${chatError.message}`
                : "I ran into an issue reaching the assistant.",
            timestamp: new Date().toISOString(),
          },
        ]);
      } finally {
        setChatLoading(false);
      }
    },
    [token, chatMessages, loadOverview],
  );

  const authError = useMemo(() => (view === "dashboard" ? null : error), [view, error]);

  if (initializingBypass || (!authConfig && !token)) {
    return (
      <div className="splash">
        <LogoMark size={56} />
        Preparing your workspace…
      </div>
    );
  }

  if (!token || !user || view !== "dashboard") {
    if (authConfig?.googleClientId) {
      return (
        <GoogleLoginPage
          clientId={authConfig.googleClientId}
          onCredential={handleGoogleCredential}
          loading={loading}
          error={authError}
        />
      );
    }
    return view === "signup" ? (
      <SignupForm
        loading={loading}
        error={authError}
        onSubmit={handleSignup}
        onSwitchToLogin={() => {
          setError(null);
          setView("login");
        }}
      />
    ) : (
      <LoginForm
        loading={loading}
        error={authError}
        onSubmit={handleLogin}
        onSwitchToSignup={() => {
          setError(null);
          setView("signup");
        }}
      />
    );
  }

  const renderContent = () => {
    switch (page) {
      case "home":
        return (
          <HomeOverview
            user={user}
            overview={overview}
            loading={loading}
            onRefresh={refreshAll}
            onRequestPrediction={handlePredictStrategy}
            latestPrediction={predictionResult}
            sparklines={sparklines}
          />
        );
      case "simulations":
        return (
          <Dashboard
            user={user}
            watchlist={watchlist}
            simulations={simulations}
            onRefreshWatchlist={refreshAll}
            onCreateSimulation={handleCreateSimulation}
            onUpdateSimulation={handleUpdateSimulation}
            onDeleteSimulation={handleDeleteSimulation}
            onTrainStrategy={handleTrainStrategy}
            onPredictStrategy={handlePredictStrategy}
            recentTraining={trainingResult}
            recentPrediction={predictionResult}
            onLogout={handleLogout}
            loading={loading || labLoading}
            error={error}
          />
        );
      case "chat":
        return <ChatbotPanel messages={chatMessages} loading={chatLoading} onSend={handleChatSend} />;
      case "strategies":
        return <StrategyCatalog strategies={strategies} />;
      case "visitors":
        return user.isAdmin ? <VisitorsPage onLoad={loadVisitors} /> : null;
      case "live":
        return (
          <LiveMarketPage
            defaultSymbols={WATCHLIST_SYMBOLS}
            onSearch={handleSearchSymbols}
            onFetchChart={handleFetchChart}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-row">
            <LogoMark size={40} />
            <strong>
              Algo Trade
              <small>Simulator</small>
            </strong>
          </div>
          <div className="profile">
            {user.picture ? (
              <img className="avatar" src={user.picture} alt="" referrerPolicy="no-referrer" />
            ) : (
              <span className="avatar placeholder">{user.name.slice(0, 1).toUpperCase()}</span>
            )}
            <div>
              <div>{user.name}</div>
              <span className="subtle">{user.email}</span>
            </div>
          </div>
        </div>
        <nav>
          <span className="nav-label">Workspace</span>
          {NAV_ITEMS.filter((item) => !item.adminOnly || user.isAdmin).map(({ page: target, label, icon: NavIcon }) => (
            <button
              key={target}
              type="button"
              className={page === target ? "active" : ""}
              aria-current={page === target ? "page" : undefined}
              onClick={() => setPage(target)}
            >
              <NavIcon />
              {label}
            </button>
          ))}
        </nav>
        <button type="button" className="logout button-ghost" onClick={handleLogout}>
          <LogoutIcon />
          Log out
        </button>
        <p className="sidebar-footnote">For education only · not financial advice</p>
      </aside>
      <main className="content">
        {error && page !== "simulations" ? <div className="error-banner">{error}</div> : null}
        {renderContent()}
      </main>
    </div>
  );
}










