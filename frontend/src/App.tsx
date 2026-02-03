import {
  SignedIn,
  SignedOut,
  SignInButton,
  SignUpButton,
  UserButton,
  OrganizationSwitcher,
} from '@clerk/clerk-react'
import './App.css'

function App() {
  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1>TechDocs AI</h1>
        </div>
        <div className="header-right">
          <SignedOut>
            <SignInButton mode="modal">
              <button className="btn btn-secondary">Sign In</button>
            </SignInButton>
            <SignUpButton mode="modal">
              <button className="btn btn-primary">Sign Up</button>
            </SignUpButton>
          </SignedOut>
          <SignedIn>
            <OrganizationSwitcher
              hidePersonal={true}
              afterCreateOrganizationUrl="/"
              afterSelectOrganizationUrl="/"
            />
            <UserButton afterSignOutUrl="/" />
          </SignedIn>
        </div>
      </header>

      <main className="main">
        <SignedOut>
          <div className="hero">
            <h2>AI-Powered Technical Documentation</h2>
            <p>Upload your manuals, query with natural language, get instant answers.</p>
            <SignUpButton mode="modal">
              <button className="btn btn-primary btn-large">Get Started</button>
            </SignUpButton>
          </div>
        </SignedOut>

        <SignedIn>
          <div className="dashboard">
            <p>Welcome! Dashboard coming soon...</p>
          </div>
        </SignedIn>
      </main>
    </div>
  )
}

export default App
