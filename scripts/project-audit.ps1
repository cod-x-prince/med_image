# ==========================================
# HemaVision AI — Project Safety Audit v1.0
# ==========================================
# Purpose:
# - Inspect project structure
# - Detect secrets safely (text files only)
# - Analyze medical/image data footprint
# - Prevent unsafe GitHub pushes
# ==========================================

Write-Host "`n=== HemaVision Project Safety Audit ===`n" -ForegroundColor Cyan

$ROOT = Get-Location

# ------------------------------------------
# 1. PROJECT STRUCTURE (Directories Only)
# ------------------------------------------
Write-Host "`n[1] PROJECT STRUCTURE (Top 3 Levels)`n" -ForegroundColor Yellow

Get-ChildItem -Path $ROOT -Directory -Recurse -Depth 3 |
ForEach-Object {
    $depth = $_.FullName.Replace($ROOT.Path, "").Split('\').Count - 1
    $indent = " " * ($depth * 2)
    Write-Host "$indent- $($_.Name)"
}

# ------------------------------------------
# 2. SECRET / TOKEN SCAN (TEXT FILES ONLY)
# ------------------------------------------
Write-Host "`n[2] SECRET / TOKEN SCAN (Safe Text Files Only)`n" -ForegroundColor Yellow

$patterns = @(
    "API_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "ACCESS_KEY",
    "PRIVATE_KEY"
)

# Only scan known text-based files under 5MB
$textFiles = Get-ChildItem -Path $ROOT -Recurse -File |
Where-Object {
    $_.Length -lt 5MB -and
    $_.Extension -match '\.(py|js|ts|html|css|json|env|yml|yaml|md|txt)$'
}

$matches = @()
foreach ($file in $textFiles) {
    $matches += Select-String -Path $file.FullName -Pattern $patterns -ErrorAction SilentlyContinue
}

if ($matches.Count -gt 0) {
    Write-Host "❌ Potential secrets found:" -ForegroundColor Red
    $matches | Select Path, LineNumber, Line | Format-Table -AutoSize
} else {
    Write-Host "✅ No obvious secrets found." -ForegroundColor Green
}

# ------------------------------------------
# 3. DANGEROUS DIRECTORIES (DATA / ENV)
# ------------------------------------------
Write-Host "`n[3] DANGEROUS DIRECTORIES CHECK`n" -ForegroundColor Yellow

$dangerDirs = @(
    "data",
    "uploads",
    "benchmark_logs",
    "benchmark_results",
    "__pycache__",
    "venv",
    ".env"
)

foreach ($dir in $dangerDirs) {
    $found = Get-ChildItem -Path $ROOT -Recurse -Directory -Filter $dir -ErrorAction SilentlyContinue
    if ($found) {
        Write-Host "❌ Found risky directory: $dir" -ForegroundColor Red
        $found | ForEach-Object {
            Write-Host "   → $($_.FullName)" -ForegroundColor DarkYellow
        }
    }
}

# ------------------------------------------
# 4. IMAGE / MEDICAL FILE ANALYSIS
# ------------------------------------------
Write-Host "`n[4] IMAGE & MEDICAL FILE ANALYSIS`n" -ForegroundColor Yellow

$imageExtensions = @(".png",".jpg",".jpeg",".dcm",".dicom",".nii",".gz")

$imageFiles = Get-ChildItem -Path $ROOT -Recurse -File |
Where-Object { $imageExtensions -contains $_.Extension }

if ($imageFiles.Count -gt 0) {

    $imageStats = $imageFiles |
    Group-Object Extension |
    ForEach-Object {
        $totalBytes = ($_.Group | Measure-Object Length -Sum).Sum
        [PSCustomObject]@{
            Extension = $_.Name
            Count     = $_.Count
            SizeMB    = [math]::Round($totalBytes / 1MB, 2)
        }
    }

    Write-Host "📊 Image / Medical File Breakdown:" -ForegroundColor Cyan
    $imageStats | Sort-Object SizeMB -Descending | Format-Table -AutoSize

    Write-Host "`n📷 Largest Image / Medical Files (>1MB):" -ForegroundColor Cyan
    $imageFiles |
        Where-Object { $_.Length -gt 1MB } |
        Sort-Object Length -Descending |
        Select-Object -First 10 |
        Select Name,
               @{n="Size(MB)";e={[math]::Round($_.Length / 1MB,2)}},
               Directory |
        Format-Table -AutoSize

    Write-Host "`nTotal image / medical files: $($imageFiles.Count)" -ForegroundColor Green
} else {
    Write-Host "✅ No image or medical files found." -ForegroundColor Green
}

# ------------------------------------------
# 5. LARGE NON-IMAGE FILES (>5MB)
# ------------------------------------------
Write-Host "`n[5] LARGE NON-IMAGE FILES (>5MB)`n" -ForegroundColor Yellow

$largeFiles = Get-ChildItem -Path $ROOT -Recurse -File |
Where-Object {
    $_.Length -gt 5MB -and
    $imageExtensions -notcontains $_.Extension
}

if ($largeFiles.Count -gt 0) {
    Write-Host "❌ Large non-image files detected:" -ForegroundColor Red
    $largeFiles |
        Sort-Object Length -Descending |
        Select Name,
               @{n="Size(MB)";e={[math]::Round($_.Length / 1MB,2)}},
               Directory |
        Format-Table -AutoSize
} else {
    Write-Host "✅ No large non-image files found." -ForegroundColor Green
}

# ------------------------------------------
# 6. REPOSITORY SIZE SUMMARY
# ------------------------------------------
Write-Host "`n[6] REPOSITORY SIZE SUMMARY`n" -ForegroundColor Yellow

$allFiles = Get-ChildItem -Path $ROOT -Recurse -File
$totalBytes = ($allFiles | Measure-Object Length -Sum).Sum
$totalMB = [math]::Round($totalBytes / 1MB, 2)

Write-Host "Total files: $($allFiles.Count)" -ForegroundColor Cyan
Write-Host "Total size : $totalMB MB" -ForegroundColor Cyan

if ($totalMB -gt 100) {
    Write-Host "`n⚠️  WARNING: Repository size exceeds 100MB" -ForegroundColor Red
    Write-Host "   Medical data must NOT be pushed to GitHub." -ForegroundColor Yellow
}

# ------------------------------------------
# 7. GIT SAFETY RECOMMENDATIONS
# ------------------------------------------
Write-Host "`n[7] GIT SAFETY RECOMMENDATIONS`n" -ForegroundColor Yellow

@"
✔ Push code only (frontend / backend source)
❌ Do NOT push medical images, datasets, or uploads

Recommended .gitignore entries:

data/
app/static/uploads/
benchmark_logs/
benchmark_results/

*.png
*.jpg
*.jpeg
*.dcm
*.dicom
*.nii
*.nii.gz

.env
venv/
__pycache__/
.vscode/
.idea/

UI images/icons should live in:
static/assets/
static/icons/
"@ | Write-Host -ForegroundColor Cyan

Write-Host "`n=== AUDIT COMPLETE ===`n" -ForegroundColor Green
