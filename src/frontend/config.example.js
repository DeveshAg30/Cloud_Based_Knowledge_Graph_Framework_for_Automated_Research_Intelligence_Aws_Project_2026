// Copy to config.js and fill in from the `sam deploy` stack outputs.
window.APP_CONFIG = {
  apiUrl: "https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/dev",        // ApiUrl
  cognitoDomain: "https://research-kg-123456789012-dev.auth.us-east-1.amazoncognito.com", // CognitoDomain
  clientId: "xxxxxxxxxxxxxxxxxxxxxxxxxx",                                       // UserPoolClientId
  region: "us-east-1",
  redirectUri: window.location.origin + "/",                                    // must match DashboardUrl
};
