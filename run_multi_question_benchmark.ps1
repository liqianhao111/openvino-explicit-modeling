#Requires -Version 5.1
<#
.SYNOPSIS
    DFlash Multi-Question Category Benchmark.

.DESCRIPTION
    Tests 10 different question categories (reasoning, knowledge, math, language
    understanding, etc.) with Baseline FP16, DFlash FP16/FP16, and DFlash FP16/INT4.
    Prints per-question acceptance stats and a final summary table.

.EXAMPLE
    .\run_multi_question_benchmark.ps1
    .\run_multi_question_benchmark.ps1 --max-tokens 256
#>
param()

$ErrorActionPreference = 'Continue'
$startTime = Get-Date

# ── Parse flags ──────────────────────────────────────────────────
$customMaxTokens = $null
for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "--max-tokens" -and ($i + 1) -lt $args.Count) {
        $customMaxTokens = [int]$args[$i + 1]
    }
}

# ── Paths ────────────────────────────────────────────────────────
$scriptDir = $PSScriptRoot
if (-not $scriptDir) { $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
$rootDir = Split-Path -Parent $scriptDir

# ── DLL environment ──────────────────────────────────────────────
$dllDirs = @(
    (Join-Path $rootDir "openvino\bin\intel64\Release"),
    (Join-Path $rootDir "openvino\temp\Windows_AMD64\tbb\bin"),
    (Join-Path $rootDir "openvino.genai\build\openvino_genai"),
    (Join-Path $rootDir "openvino.genai\build\bin\Release")
)
$extraPath = ($dllDirs | Where-Object { Test-Path $_ }) -join ";"
$env:PATH = "$extraPath;$($env:PATH)"
$env:OV_GENAI_USE_MODELING_API = "1"
$env:OV_GENAI_DISABLE_THINKING = "1"

$baselineExe = Join-Path $rootDir "openvino.genai\build\bin\Release\modeling_qwen3_5.exe"
$dflashExe   = Join-Path $rootDir "openvino.genai\build\bin\Release\modeling_qwen3_5_dflash.exe"

if (-not (Test-Path $baselineExe)) { Write-Error "Baseline exe not found: $baselineExe"; exit 1 }
if (-not (Test-Path $dflashExe))   { Write-Error "DFlash exe not found: $dflashExe"; exit 1 }

$modelDir  = "D:\Data\models\Huggingface\Qwen3.5-4B"
$draftDir  = "D:\Data\models\Huggingface\Qwen3.5-4B-DFlash"
$device    = "GPU"
$maxTokens = if ($customMaxTokens) { $customMaxTokens } else { 256 }
$blockSize = 16
$reportDir = Join-Path $rootDir "dflash_multi_question_reports"

# ═════════════════════════════════════════════════════════════════
# 10 Question Categories
# ═════════════════════════════════════════════════════════════════
$questions = @(
    @{
        Category = "Logical Reasoning"
        Prompt   = "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Explain your reasoning step by step."
    },
    @{
        Category = "Factual Knowledge"
        Prompt   = "What is the capital of Australia, and why do many people incorrectly think it is Sydney? Provide historical context."
    },
    @{
        Category = "Math Calculation"
        Prompt   = "Calculate the result of 347 multiplied by 28, then subtract 1523. Show your work step by step."
    },
    @{
        Category = "Reading Comprehension"
        Prompt   = "Read the following passage and answer the question. Passage: 'The Great Wall of China, built over many centuries, stretches approximately 13,171 miles. Contrary to popular belief, it is not visible from space with the naked eye. The wall was primarily built to protect against invasions from northern nomadic groups.' Question: What is a common misconception about the Great Wall mentioned in this passage?"
    },
    @{
        Category = "Creative Writing"
        Prompt   = "Write a short poem (8 lines) about the beauty of sunrise over the ocean. Use vivid imagery and at least one metaphor."
    },
    @{
        Category = "Code Generation"
        Prompt   = "Write a Python function called 'fibonacci' that takes an integer n and returns the nth Fibonacci number using dynamic programming. Include a brief docstring."
    },
    @{
        Category = "Causal Reasoning"
        Prompt   = "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain why or why not using principles of formal logic."
    },
    @{
        Category = "Summarization"
        Prompt   = "Summarize the following in 2-3 sentences: 'Artificial intelligence has transformed industries ranging from healthcare to finance. In healthcare, AI assists in diagnosing diseases, predicting patient outcomes, and personalizing treatment plans. In finance, AI algorithms detect fraud, automate trading, and assess credit risk. Despite these advances, concerns about job displacement, bias in algorithms, and data privacy continue to spark debate among policymakers, technologists, and the public.'"
    },
    @{
        Category = "Translation & Multilingual"
        Prompt   = "Translate the following English sentence into French, Spanish, and German: 'The quick brown fox jumps over the lazy dog.' Then explain any interesting linguistic differences between the three translations."
    },
    @{
        Category = "Commonsense & Analogy"
        Prompt   = "Complete the analogy: 'Book is to reading as fork is to ___.' Explain your reasoning and provide two more analogies following the same pattern."
    }
)

# ═════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════

function Parse-ExeOutput([string]$stdout) {
    $m = @{
        Label = ""; IsDFlash = $false
        TargetQuant = "FP16"; DraftQuant = "-"
        Tokens = 0; TTFT = 0.0; TPOT = 0.0; E2E = 0.0
        DraftSteps = 0; AcceptedDraft = 0; AcceptRate = 0.0; AvgAccepted = 0.0
        AcceptedPerStep = ""
        OutputText = ""
    }
    if ($stdout -match 'Output token size:\s*(\d+)')       { $m.Tokens = [int]$Matches[1] }
    if ($stdout -match 'TTFT:\s*([\d.]+)\s*ms')            { $m.TTFT = [double]$Matches[1] }
    $decodeMs = 0.0
    if ($stdout -match 'Decode time:\s*([\d.]+)\s*ms')     { $decodeMs = [double]$Matches[1] }
    if ($stdout -match 'TPOT:\s*([\d.]+)\s*ms/token')      { $m.TPOT = [double]$Matches[1] }
    $m.E2E = $m.TTFT + $decodeMs

    # DFlash-specific
    if ($stdout -match 'Draft steps:\s*(\d+)')              { $m.DraftSteps = [int]$Matches[1] }
    if ($stdout -match 'Accepted draft tokens:\s*(\d+)')    { $m.AcceptedDraft = [int]$Matches[1] }
    if ($stdout -match 'Acceptance rate:\s*([\d.]+)')        { $m.AcceptRate = [double]$Matches[1] }
    if ($stdout -match 'Avg accepted per step:\s*([\d.]+)')  { $m.AvgAccepted = [double]$Matches[1] }
    if ($stdout -match 'Accepted per step:\s*\[([^\]]+)\]') { $m.AcceptedPerStep = $Matches[1] }

    # Output text
    if ($stdout -match '(?s)\[Output\]\r?\n(.*?)\r?\n\r?\n\[Generation Complete\]') {
        $m.OutputText = $Matches[1].Trim()
    } else {
        $lines = $stdout -split '\r?\n'
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match '^Throughput:') {
                $remaining = ($lines[($i+1)..($lines.Count-1)] -join "`n").Trim()
                if ($remaining) { $m.OutputText = $remaining }
                break
            }
        }
    }
    return $m
}

function Invoke-Baseline {
    param([string]$Prompt, [int]$MaxTok)

    $exeArgs = @("--model", $modelDir, "--prompt", $Prompt,
                 "--device", $device, "--output-tokens", "$MaxTok",
                 "--temperature", "0", "--think", "0", "--mode", "text")

    try {
        $output = & $baselineExe @exeArgs 2>&1 | Out-String
    } catch {
        Write-Host "  [Baseline FAILED] $_" -ForegroundColor Red
        return $null
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [Baseline WARNING] exit code $LASTEXITCODE" -ForegroundColor Yellow
    }

    $m = Parse-ExeOutput $output
    $m.Label = "Baseline FP16"
    $m.IsDFlash = $false
    $m.TargetQuant = "FP16"
    $m.DraftQuant = "-"
    return $m
}

function Invoke-DFlash {
    param([string]$Prompt, [int]$MaxTok,
          [string]$TargetQuant = "FP16", [string]$DraftQuant = "FP16")

    $exeArgs = @($modelDir, $draftDir, $Prompt, $device,
                 "$MaxTok", "$blockSize", $TargetQuant, $DraftQuant)

    try {
        $output = & $dflashExe @exeArgs 2>&1 | Out-String
    } catch {
        Write-Host "  [DFlash FAILED] $_" -ForegroundColor Red
        return $null
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [DFlash WARNING] exit code $LASTEXITCODE" -ForegroundColor Yellow
    }

    $tqLabel = if ($TargetQuant -match "INT4") { "INT4" } else { "FP16" }
    $dqLabel = if ($DraftQuant -match "INT4") { "INT4" } else { "FP16" }

    $m = Parse-ExeOutput $output
    $m.Label = "DFlash $tqLabel/$dqLabel"
    $m.IsDFlash = $true
    $m.TargetQuant = $tqLabel
    $m.DraftQuant = $dqLabel
    return $m
}

# ═════════════════════════════════════════════════════════════════
# Main Benchmark Loop
# ═════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host ("=" * 105)
Write-Host "  DFlash Multi-Question Category Benchmark"
Write-Host "  Configs: Baseline FP16, DFlash FP16/FP16, DFlash FP16/INT4"
Write-Host "  Questions: $($questions.Count)   max_tokens=$maxTokens   device=$device"
Write-Host ("=" * 105)
Write-Host ""

# Store all results: array of hashtables
# Each entry: { Category, Baseline, DFlashFP16, DFlashINT4 }
$allResults = @()

for ($qi = 0; $qi -lt $questions.Count; $qi++) {
    $q = $questions[$qi]
    $cat = $q.Category
    $prompt = $q.Prompt
    $qNum = $qi + 1

    Write-Host ""
    Write-Host ("=" * 105)
    Write-Host ("  [{0}/{1}] Category: {2}" -f $qNum, $questions.Count, $cat)
    Write-Host ("=" * 105)
    Write-Host "  Prompt: $($prompt.Substring(0, [Math]::Min(100, $prompt.Length)))..."
    Write-Host ""

    # ── Baseline FP16 ──
    Write-Host "  Running Baseline FP16..."
    $baseline = Invoke-Baseline -Prompt $prompt -MaxTok $maxTokens
    if ($baseline) {
        Write-Host ("    TTFT={0:F1}ms  TPOT={1:F2}ms  Tokens={2}" -f $baseline.TTFT, $baseline.TPOT, $baseline.Tokens)
    }

    # ── DFlash FP16/FP16 ──
    Write-Host "  Running DFlash FP16/FP16..."
    $dflashFP16 = Invoke-DFlash -Prompt $prompt -MaxTok $maxTokens -TargetQuant "FP16" -DraftQuant "FP16"
    if ($dflashFP16) {
        Write-Host ("    TTFT={0:F1}ms  TPOT={1:F2}ms  Tokens={2}  Accept={3:F1}%  AvgAcc={4:F2}" -f `
            $dflashFP16.TTFT, $dflashFP16.TPOT, $dflashFP16.Tokens, ($dflashFP16.AcceptRate * 100), $dflashFP16.AvgAccepted)
    }

    # ── DFlash FP16/INT4 ──
    Write-Host "  Running DFlash FP16/INT4..."
    $dflashINT4 = Invoke-DFlash -Prompt $prompt -MaxTok $maxTokens -TargetQuant "FP16" -DraftQuant "INT4_ASYM"
    if ($dflashINT4) {
        Write-Host ("    TTFT={0:F1}ms  TPOT={1:F2}ms  Tokens={2}  Accept={3:F1}%  AvgAcc={4:F2}" -f `
            $dflashINT4.TTFT, $dflashINT4.TPOT, $dflashINT4.Tokens, ($dflashINT4.AcceptRate * 100), $dflashINT4.AvgAccepted)
    }

    $entry = @{
        QNum      = $qNum
        Category  = $cat
        Prompt    = $prompt
        Baseline  = $baseline
        DFlashFP16 = $dflashFP16
        DFlashINT4 = $dflashINT4
    }
    $allResults += $entry
}

# ═════════════════════════════════════════════════════════════════
# Summary Tables
# ═════════════════════════════════════════════════════════════════

$sep = "=" * 145
$dash = "-" * 145

Write-Host ""
Write-Host ""
Write-Host $sep
Write-Host "  MULTI-QUESTION BENCHMARK SUMMARY"
Write-Host $sep
Write-Host ""

# ── Table 1: Performance Overview ──
Write-Host "  Table 1: Performance Overview (TPOT in ms/token, Speedup = Baseline_TPOT / DFlash_TPOT)"
Write-Host $dash
Write-Host ("{0,-4} {1,-24} {2,>8} {3,>8} {4,>8} {5,>8} {6,>8} {7,>8} {8,>8} {9,>8} {10,>8} {11,>8}" -f `
    "#", "Category",
    "BL_TTFT", "BL_TPOT", "BL_Tok",
    "FP_TPOT", "FP_Spd", "FP_Tok",
    "I4_TPOT", "I4_Spd", "I4_Tok", "TextMatch")
Write-Host $dash

foreach ($r in $allResults) {
    $blTpot  = if ($r.Baseline)    { "{0:F2}" -f $r.Baseline.TPOT }   else { "N/A" }
    $blTtft  = if ($r.Baseline)    { "{0:F0}" -f $r.Baseline.TTFT }   else { "N/A" }
    $blTok   = if ($r.Baseline)    { $r.Baseline.Tokens }             else { "N/A" }
    $fpTpot  = if ($r.DFlashFP16)  { "{0:F2}" -f $r.DFlashFP16.TPOT } else { "N/A" }
    $fpTok   = if ($r.DFlashFP16)  { $r.DFlashFP16.Tokens }           else { "N/A" }
    $i4Tpot  = if ($r.DFlashINT4)  { "{0:F2}" -f $r.DFlashINT4.TPOT } else { "N/A" }
    $i4Tok   = if ($r.DFlashINT4)  { $r.DFlashINT4.Tokens }           else { "N/A" }

    $fpSpd = "N/A"
    if ($r.Baseline -and $r.DFlashFP16 -and $r.Baseline.TPOT -gt 0 -and $r.DFlashFP16.TPOT -gt 0) {
        $fpSpd = "{0:F2}x" -f ($r.Baseline.TPOT / $r.DFlashFP16.TPOT)
    }
    $i4Spd = "N/A"
    if ($r.Baseline -and $r.DFlashINT4 -and $r.Baseline.TPOT -gt 0 -and $r.DFlashINT4.TPOT -gt 0) {
        $i4Spd = "{0:F2}x" -f ($r.Baseline.TPOT / $r.DFlashINT4.TPOT)
    }

    # Text match: DFlash FP16 vs Baseline
    $textMatch = "N/A"
    if ($r.Baseline -and $r.DFlashFP16) {
        if ($r.Baseline.OutputText -eq $r.DFlashFP16.OutputText) {
            $textMatch = "MATCH"
        } else {
            $a = $r.Baseline.OutputText; $b = $r.DFlashFP16.OutputText
            $maxLen = [Math]::Max($a.Length, $b.Length)
            if ($maxLen -eq 0) { $textMatch = "MATCH" }
            else {
                $common = 0
                $minLen = [Math]::Min($a.Length, $b.Length)
                for ($j = 0; $j -lt $minLen; $j++) { if ($a[$j] -eq $b[$j]) { $common++ } }
                $sim = [double]$common / $maxLen * 100
                $textMatch = "{0:F0}%" -f $sim
            }
        }
    }

    Write-Host ("{0,-4} {1,-24} {2,>8} {3,>8} {4,>8} {5,>8} {6,>8} {7,>8} {8,>8} {9,>8} {10,>8} {11,>8}" -f `
        $r.QNum, $r.Category,
        $blTtft, $blTpot, $blTok,
        $fpTpot, $fpSpd, $fpTok,
        $i4Tpot, $i4Spd, $i4Tok, $textMatch)
}
Write-Host $dash
Write-Host ""

# ── Table 2: DFlash Acceptance Details ──
Write-Host ""
Write-Host "  Table 2: DFlash Acceptance Rate Details"
Write-Host $dash
Write-Host ("{0,-4} {1,-24} {2,-12} {3,>8} {4,>10} {5,>10} {6,>10}   {7}" -f `
    "#", "Category", "Config", "Steps", "Accepted", "Accept%", "AvgAcc", "Per-Step Accepted")
Write-Host $dash

foreach ($r in $allResults) {
    # DFlash FP16/FP16 row
    if ($r.DFlashFP16) {
        $d = $r.DFlashFP16
        $perStep = if ($d.AcceptedPerStep) { "[$($d.AcceptedPerStep)]" } else { "-" }
        # Truncate per-step if too long
        if ($perStep.Length -gt 60) { $perStep = $perStep.Substring(0, 57) + "..." }
        Write-Host ("{0,-4} {1,-24} {2,-12} {3,>8} {4,>10} {5,>9}% {6,>10}   {7}" -f `
            $r.QNum, $r.Category, "FP16/FP16",
            $d.DraftSteps, $d.AcceptedDraft,
            ("{0:F1}" -f ($d.AcceptRate * 100)),
            ("{0:F2}" -f $d.AvgAccepted),
            $perStep)
    }
    # DFlash FP16/INT4 row
    if ($r.DFlashINT4) {
        $d = $r.DFlashINT4
        $perStep = if ($d.AcceptedPerStep) { "[$($d.AcceptedPerStep)]" } else { "-" }
        if ($perStep.Length -gt 60) { $perStep = $perStep.Substring(0, 57) + "..." }
        Write-Host ("{0,-4} {1,-24} {2,-12} {3,>8} {4,>10} {5,>9}% {6,>10}   {7}" -f `
            "", "", "FP16/INT4",
            $d.DraftSteps, $d.AcceptedDraft,
            ("{0:F1}" -f ($d.AcceptRate * 100)),
            ("{0:F2}" -f $d.AvgAccepted),
            $perStep)
    }
    Write-Host ""
}
Write-Host $dash

# ── Averages ──
Write-Host ""
Write-Host "  Overall Averages:"
$avgBLTpot = 0.0; $avgFPTpot = 0.0; $avgI4Tpot = 0.0
$avgFPAccept = 0.0; $avgI4Accept = 0.0
$avgFPAvgAcc = 0.0; $avgI4AvgAcc = 0.0
$avgFPSpeedup = 0.0; $avgI4Speedup = 0.0
$countBL = 0; $countFP = 0; $countI4 = 0

foreach ($r in $allResults) {
    if ($r.Baseline -and $r.Baseline.TPOT -gt 0) {
        $avgBLTpot += $r.Baseline.TPOT; $countBL++
    }
    if ($r.DFlashFP16 -and $r.DFlashFP16.TPOT -gt 0) {
        $avgFPTpot += $r.DFlashFP16.TPOT
        $avgFPAccept += $r.DFlashFP16.AcceptRate
        $avgFPAvgAcc += $r.DFlashFP16.AvgAccepted
        if ($r.Baseline -and $r.Baseline.TPOT -gt 0) {
            $avgFPSpeedup += ($r.Baseline.TPOT / $r.DFlashFP16.TPOT)
        }
        $countFP++
    }
    if ($r.DFlashINT4 -and $r.DFlashINT4.TPOT -gt 0) {
        $avgI4Tpot += $r.DFlashINT4.TPOT
        $avgI4Accept += $r.DFlashINT4.AcceptRate
        $avgI4AvgAcc += $r.DFlashINT4.AvgAccepted
        if ($r.Baseline -and $r.Baseline.TPOT -gt 0) {
            $avgI4Speedup += ($r.Baseline.TPOT / $r.DFlashINT4.TPOT)
        }
        $countI4++
    }
}

if ($countBL -gt 0) { $avgBLTpot /= $countBL }
if ($countFP -gt 0) { $avgFPTpot /= $countFP; $avgFPAccept /= $countFP; $avgFPAvgAcc /= $countFP; $avgFPSpeedup /= $countFP }
if ($countI4 -gt 0) { $avgI4Tpot /= $countI4; $avgI4Accept /= $countI4; $avgI4AvgAcc /= $countI4; $avgI4Speedup /= $countI4 }

Write-Host ("    Baseline FP16     avg TPOT = {0:F2} ms/token" -f $avgBLTpot)
Write-Host ("    DFlash FP16/FP16  avg TPOT = {0:F2} ms/token  avg Speedup = {1:F2}x  avg Accept = {2:F1}%  avg AvgAcc = {3:F2}" -f `
    $avgFPTpot, $avgFPSpeedup, ($avgFPAccept * 100), $avgFPAvgAcc)
Write-Host ("    DFlash FP16/INT4  avg TPOT = {0:F2} ms/token  avg Speedup = {1:F2}x  avg Accept = {2:F1}%  avg AvgAcc = {3:F2}" -f `
    $avgI4Tpot, $avgI4Speedup, ($avgI4Accept * 100), $avgI4AvgAcc)

# ═════════════════════════════════════════════════════════════════
# Save Report
# ═════════════════════════════════════════════════════════════════

if (-not (Test-Path $reportDir)) { New-Item -ItemType Directory -Path $reportDir -Force | Out-Null }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$reportPath = Join-Path $reportDir "multi_question_benchmark_${ts}.txt"

$reportLines = @()
$reportLines += "DFlash Multi-Question Category Benchmark Report"
$reportLines += "=" * 80
$reportLines += "timestamp=$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')"
$reportLines += "model_dir=$modelDir"
$reportLines += "draft_dir=$draftDir"
$reportLines += "device=$device"
$reportLines += "max_tokens=$maxTokens"
$reportLines += "block_size=$blockSize"
$reportLines += "questions=$($questions.Count)"
$reportLines += ""

# Per-question details
for ($qi = 0; $qi -lt $allResults.Count; $qi++) {
    $r = $allResults[$qi]
    $reportLines += "=" * 80
    $reportLines += "[Q$($r.QNum)] $($r.Category)"
    $reportLines += "Prompt: $($r.Prompt)"
    $reportLines += ""

    if ($r.Baseline) {
        $b = $r.Baseline
        $reportLines += "[Baseline FP16]"
        $reportLines += "  TTFT={0:F1}ms  TPOT={1:F2}ms  E2E={2:F1}ms  Tokens={3}" -f $b.TTFT, $b.TPOT, $b.E2E, $b.Tokens
        if ($b.OutputText) {
            $reportLines += "  [Output]"
            $reportLines += $b.OutputText
            $reportLines += ""
        }
    }

    if ($r.DFlashFP16) {
        $d = $r.DFlashFP16
        $reportLines += "[DFlash FP16/FP16]"
        $reportLines += "  TTFT={0:F1}ms  TPOT={1:F2}ms  E2E={2:F1}ms  Tokens={3}" -f $d.TTFT, $d.TPOT, $d.E2E, $d.Tokens
        $reportLines += "  DraftSteps={0}  AcceptedDraft={1}  AcceptRate={2:F4}  AvgAccepted={3:F4}" -f `
            $d.DraftSteps, $d.AcceptedDraft, $d.AcceptRate, $d.AvgAccepted
        if ($d.AcceptedPerStep) { $reportLines += "  AcceptedPerStep=[$($d.AcceptedPerStep)]" }
        if ($d.OutputText) {
            $reportLines += "  [Output]"
            $reportLines += $d.OutputText
            $reportLines += ""
        }
        # Text match
        if ($r.Baseline) {
            if ($r.Baseline.OutputText -eq $d.OutputText) {
                $reportLines += "  TextMatch: IDENTICAL"
            } else {
                $reportLines += "  TextMatch: DIFFERENT"
            }
        }
    }

    if ($r.DFlashINT4) {
        $d = $r.DFlashINT4
        $reportLines += "[DFlash FP16/INT4]"
        $reportLines += "  TTFT={0:F1}ms  TPOT={1:F2}ms  E2E={2:F1}ms  Tokens={3}" -f $d.TTFT, $d.TPOT, $d.E2E, $d.Tokens
        $reportLines += "  DraftSteps={0}  AcceptedDraft={1}  AcceptRate={2:F4}  AvgAccepted={3:F4}" -f `
            $d.DraftSteps, $d.AcceptedDraft, $d.AcceptRate, $d.AvgAccepted
        if ($d.AcceptedPerStep) { $reportLines += "  AcceptedPerStep=[$($d.AcceptedPerStep)]" }
        if ($d.OutputText) {
            $reportLines += "  [Output]"
            $reportLines += $d.OutputText
            $reportLines += ""
        }
        if ($r.Baseline) {
            if ($r.Baseline.OutputText -eq $d.OutputText) {
                $reportLines += "  TextMatch: IDENTICAL"
            } else {
                $reportLines += "  TextMatch: DIFFERENT"
            }
        }
    }
    $reportLines += ""
}

# Summary tables (same as console output)
$reportLines += ""
$reportLines += $sep
$reportLines += "  SUMMARY TABLE 1: Performance Overview"
$reportLines += $sep
$reportLines += ("{0,-4} {1,-24} {2,>8} {3,>8} {4,>8} {5,>8} {6,>8} {7,>8} {8,>8} {9,>8} {10,>8}" -f `
    "#", "Category", "BL_TTFT", "BL_TPOT", "BL_Tok", "FP_TPOT", "FP_Spd", "FP_Tok", "I4_TPOT", "I4_Spd", "I4_Tok")
$reportLines += $dash

foreach ($r in $allResults) {
    $blTpot  = if ($r.Baseline)    { "{0:F2}" -f $r.Baseline.TPOT }   else { "N/A" }
    $blTtft  = if ($r.Baseline)    { "{0:F0}" -f $r.Baseline.TTFT }   else { "N/A" }
    $blTok   = if ($r.Baseline)    { $r.Baseline.Tokens }             else { "N/A" }
    $fpTpot  = if ($r.DFlashFP16)  { "{0:F2}" -f $r.DFlashFP16.TPOT } else { "N/A" }
    $fpTok   = if ($r.DFlashFP16)  { $r.DFlashFP16.Tokens }           else { "N/A" }
    $i4Tpot  = if ($r.DFlashINT4)  { "{0:F2}" -f $r.DFlashINT4.TPOT } else { "N/A" }
    $i4Tok   = if ($r.DFlashINT4)  { $r.DFlashINT4.Tokens }           else { "N/A" }

    $fpSpd = "N/A"
    if ($r.Baseline -and $r.DFlashFP16 -and $r.Baseline.TPOT -gt 0 -and $r.DFlashFP16.TPOT -gt 0) {
        $fpSpd = "{0:F2}x" -f ($r.Baseline.TPOT / $r.DFlashFP16.TPOT)
    }
    $i4Spd = "N/A"
    if ($r.Baseline -and $r.DFlashINT4 -and $r.Baseline.TPOT -gt 0 -and $r.DFlashINT4.TPOT -gt 0) {
        $i4Spd = "{0:F2}x" -f ($r.Baseline.TPOT / $r.DFlashINT4.TPOT)
    }

    $reportLines += ("{0,-4} {1,-24} {2,>8} {3,>8} {4,>8} {5,>8} {6,>8} {7,>8} {8,>8} {9,>8} {10,>8}" -f `
        $r.QNum, $r.Category, $blTtft, $blTpot, $blTok, $fpTpot, $fpSpd, $fpTok, $i4Tpot, $i4Spd, $i4Tok)
}
$reportLines += $dash

$reportLines += ""
$reportLines += "  SUMMARY TABLE 2: DFlash Acceptance Rate Details"
$reportLines += $dash
$reportLines += ("{0,-4} {1,-24} {2,-12} {3,>8} {4,>10} {5,>10} {6,>10}   {7}" -f `
    "#", "Category", "Config", "Steps", "Accepted", "Accept%", "AvgAcc", "Per-Step Accepted")
$reportLines += $dash

foreach ($r in $allResults) {
    if ($r.DFlashFP16) {
        $d = $r.DFlashFP16
        $perStep = if ($d.AcceptedPerStep) { "[$($d.AcceptedPerStep)]" } else { "-" }
        $reportLines += ("{0,-4} {1,-24} {2,-12} {3,>8} {4,>10} {5,>9}% {6,>10}   {7}" -f `
            $r.QNum, $r.Category, "FP16/FP16",
            $d.DraftSteps, $d.AcceptedDraft,
            ("{0:F1}" -f ($d.AcceptRate * 100)),
            ("{0:F2}" -f $d.AvgAccepted),
            $perStep)
    }
    if ($r.DFlashINT4) {
        $d = $r.DFlashINT4
        $perStep = if ($d.AcceptedPerStep) { "[$($d.AcceptedPerStep)]" } else { "-" }
        $reportLines += ("{0,-4} {1,-24} {2,-12} {3,>8} {4,>10} {5,>10} {6,>10}   {7}" -f `
            "", "", "FP16/INT4",
            $d.DraftSteps, $d.AcceptedDraft,
            ("{0:F1}" -f ($d.AcceptRate * 100)),
            ("{0:F2}" -f $d.AvgAccepted),
            $perStep)
    }
    $reportLines += ""
}
$reportLines += $dash

$reportLines += ""
$reportLines += "  Overall Averages:"
$reportLines += ("    Baseline FP16     avg TPOT = {0:F2} ms/token" -f $avgBLTpot)
$reportLines += ("    DFlash FP16/FP16  avg TPOT = {0:F2} ms/token  avg Speedup = {1:F2}x  avg Accept = {2:F1}%  avg AvgAcc = {3:F2}" -f `
    $avgFPTpot, $avgFPSpeedup, ($avgFPAccept * 100), $avgFPAvgAcc)
$reportLines += ("    DFlash FP16/INT4  avg TPOT = {0:F2} ms/token  avg Speedup = {1:F2}x  avg Accept = {2:F1}%  avg AvgAcc = {3:F2}" -f `
    $avgI4Tpot, $avgI4Speedup, ($avgI4Accept * 100), $avgI4AvgAcc)

$reportLines | Out-File -FilePath $reportPath -Encoding UTF8
Write-Host ""
Write-Host "Report saved to: $reportPath"

$elapsed = (Get-Date) - $startTime
Write-Host ("Total elapsed: {0:F1} minutes" -f $elapsed.TotalMinutes)
