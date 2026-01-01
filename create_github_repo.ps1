# Script to create GitHub repository and push code
# This script requires GitHub Personal Access Token

param(
    [Parameter(Mandatory=$false)]
    [string]$RepoName = "ROM_RL_Geothermal",
    
    [Parameter(Mandatory=$false)]
    [string]$GitHubUsername = "",
    
    [Parameter(Mandatory=$false)]
    [string]$GitHubToken = ""
)

# If token not provided, try to get from environment
if ([string]::IsNullOrEmpty($GitHubToken)) {
    $GitHubToken = $env:GITHUB_TOKEN
}

# If username not provided, try to infer from git config
if ([string]::IsNullOrEmpty($GitHubUsername)) {
    $GitHubUsername = (git config user.name)
    if ([string]::IsNullOrEmpty($GitHubUsername)) {
        Write-Host "Please provide GitHub username: " -NoNewline
        $GitHubUsername = Read-Host
    }
}

# If token still not available, prompt
if ([string]::IsNullOrEmpty($GitHubToken)) {
    Write-Host "GitHub Personal Access Token is required to create the repository."
    Write-Host "You can create one at: https://github.com/settings/tokens"
    Write-Host "Please enter your GitHub Personal Access Token: " -NoNewline
    $GitHubToken = Read-Host -AsSecureString
    $GitHubToken = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($GitHubToken))
}

# Create repository via GitHub API
$headers = @{
    "Authorization" = "token $GitHubToken"
    "Accept" = "application/vnd.github.v3+json"
}

$body = @{
    name = $RepoName
    description = "ROM and RL Geothermal Project"
    private = $false
} | ConvertTo-Json

try {
    Write-Host "Creating GitHub repository: $RepoName..."
    $response = Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Method Post -Headers $headers -Body $body -ContentType "application/json"
    
    $repoUrl = $response.clone_url
    Write-Host "Repository created successfully: $repoUrl"
    
    # Add remote and push
    Write-Host "Adding remote origin..."
    git remote add origin $repoUrl 2>$null
    if ($LASTEXITCODE -ne 0) {
        git remote set-url origin $repoUrl
    }
    
    Write-Host "Pushing code to GitHub..."
    git branch -M main
    git push -u origin main
    
    Write-Host "Successfully pushed code to GitHub!"
    Write-Host "Repository URL: $($response.html_url)"
} catch {
    Write-Host "Error creating repository: $_" -ForegroundColor Red
    Write-Host "You may need to create the repository manually on GitHub.com"
    Write-Host "Then run: git remote add origin https://github.com/$GitHubUsername/$RepoName.git"
    Write-Host "And: git push -u origin main"
}

