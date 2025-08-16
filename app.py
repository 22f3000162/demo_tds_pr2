import os
import sys
import subprocess
import tempfile
import json
import shutil
import re
import logging
import time
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI,UploadFile, File
from fastapi.responses import JSONResponse
from openai import OpenAI

# =========================
# Setup
# =========================
load_dotenv()

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Analyst API", version="1.0")


# =========================
# PureLLMAnalyst class (cut-down version)
# =========================

class PureLLMAnalyst:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openrouter/v1")
        )
        self.model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
        self.max_retries = 3
        self.timeout = 120  # Increased timeout for DuckDB operations
        
        # Enhanced detection patterns - Fixed and improved
        self.duckdb_patterns = [
            r'duckdb|duck\s*db|SELECT.*FROM.*read_parquet',
            r's3://[^\s\)]+\.(parquet|csv)',
            r'INSTALL\s+(httpfs|parquet)|LOAD\s+(httpfs|parquet)',
            r'read_parquet\s*\(',
            r'\$\d+[mM].*records|\d+[tT][bB].*data',
            r'metadata\.parquet|s3_region='
        ]
        
        self.web_patterns = [
            r'scrape|fetch|crawl|extract.*from.*url',
            r'wikipedia\.org|github\.com|reddit\.com',
            r'https?://(?!.*s3://)[^\s\)]+\.(?:com|org|edu|gov|net|io)(?!/[^\s]*\.(?:parquet|csv))'
        ]
        
        self.csv_patterns = [
            r'csv|dataframe|table.*file|local.*file',
            r'columns?|rows?.*data|analyze.*csv'
        ]
        
        self.viz_patterns = [
            r'plot|chart|graph|visualization|scatter|bar|line|histogram',
            r'base64|image|png|jpg|matplotlib|seaborn|plotly',
            r'regression.*line|scatterplot'
        ]

    def detect_and_prepare_context(self, question: str, **kwargs) -> Dict[str, Any]:
        """Enhanced LLM analysis with better DuckDB detection"""
        logger.info("🧠 LLM analyzing question for data sources and requirements...")
        
        try:
            detection_prompt = f"""Analyze this data analysis question and determine the PRIMARY data source and workflow:

Question: {question}

CRITICAL DETECTION RULES:
1. If contains S3 parquet paths (s3://bucket/path/*.parquet) + DuckDB queries → CLOUD_DUCKDB (highest priority)
2. If contains valid web URLs (NOT references/citations) that need scraping → WEB_SCRAPING
3. If contains local CSV files or uploaded data → CSV_ANALYSIS
4. If contains mixed sources → MIXED

SPECIFIC INDICATORS:
- DuckDB/Cloud: "duckdb", "s3://", "read_parquet", "INSTALL httpfs", "LOAD httpfs", "s3_region="
- Web Scraping: "scrape", "fetch from URL", actual clickable URLs like "https://domain.com"
- CSV Analysis: ".csv files", "dataframe", "local data", "uploaded files"

EXTRACT CAREFULLY:
- S3 paths: s3://bucket/path/file.parquet
- SQL queries: Complete SELECT statements
- Valid web URLs: Only actual URLs to scrape (not citations)
- File references: Local file paths

DO NOT EXTRACT:
- Citation links like [ecourts website](https://judgments.ecourts.gov.in/) - these are references, not scrape targets
- URLs that are just mentioned as data source descriptions
- Malformed URLs with extra characters

Return JSON analysis:
{{
    "primary_data_source": "cloud_duckdb|web_scraping|csv_analysis|mixed",
    "extracted_s3_paths": ["s3://bucket/path/file.parquet", ...],
    "extracted_sql_queries": ["SELECT...", ...],
    "extracted_web_urls": ["https://actual-scrape-target.com", ...],
    "extracted_local_files": ["file.csv", ...],
    "needs_duckdb": true/false,
    "needs_web_scraping": true/false,
    "needs_csv_analysis": true/false,
    "needs_visualization": true/false,
    "needs_statistical_analysis": true/false,
    "data_scale": "small|medium|large|massive",
    "analysis_complexity": "simple|medium|complex",
    "expected_output": "json|text|image|mixed",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of detection logic",
    "key_technologies": ["duckdb", "pandas", "matplotlib", "etc"],
    "workflow_steps": ["step1", "step2", "step3"],
    "cloud_config": {{"region": "ap-south-1", "requires_credentials": false}},
    "citation_links_found": ["citation1", "citation2"],
    "actual_scrape_targets": ["url1", "url2"]
}}

EXAMPLE ANALYSIS:
- "DuckDB query on s3://bucket/*.parquet" → primary_data_source: "cloud_duckdb", needs_duckdb: true
- "Scrape https://wikipedia.org/wiki/Films and analyze" → primary_data_source: "web_scraping", needs_web_scraping: true
- "Dataset from [source](url)" → citation_links_found: [url], actual_scrape_targets: []"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": detection_prompt}],
                temperature=0.1,
                timeout=20
            )
            
            # Safe JSON parsing
            response_content = response.choices[0].message.content.strip()
            try:
                analysis = json.loads(response_content)
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON decode error in LLM response: {e}")
                logger.info("🔄 Falling back to pattern-based detection...")
                return self._fallback_detection(question, **kwargs)
            
            # Log detailed detection results
            primary_source = analysis.get('primary_data_source', 'unknown')
            logger.info(f"🎯 Primary data source detected: {primary_source}")
            logger.info(f"📊 Confidence: {analysis.get('confidence', 'unknown')}")
            
            # Log what was found
            if analysis.get('extracted_s3_paths'):
                logger.info(f"☁️ S3 paths detected: {len(analysis['extracted_s3_paths'])} paths")
            if analysis.get('extracted_sql_queries'):
                logger.info(f"🔧 SQL queries detected: {len(analysis['extracted_sql_queries'])} queries")
            if analysis.get('extracted_web_urls'):
                logger.info(f"🌐 Web URLs for scraping: {analysis['extracted_web_urls']}")
            if analysis.get('citation_links_found'):
                logger.info(f"📄 Citation links (not for scraping): {analysis['citation_links_found']}")
            
            # Validate detection quality
            if analysis.get('confidence', 0) < 0.6:
                logger.warning("🔄 Low confidence detection, using enhanced fallback...")
                return self._fallback_detection(question, **kwargs)
            
            # Override web scraping if no actual scrape targets
            if (analysis.get('needs_web_scraping') and 
                not analysis.get('actual_scrape_targets') and 
                analysis.get('extracted_s3_paths')):
                logger.info("🚫 Disabling web scraping - only citations found, S3 data available")
                analysis['needs_web_scraping'] = False
                analysis['primary_data_source'] = 'cloud_duckdb'
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error in LLM extraction: {e}")
            logger.info("🔄 Falling back to pattern-based detection...")
            return self._fallback_detection(question, **kwargs)
    
    def _fallback_detection(self, question: str, **kwargs) -> Dict[str, Any]:
        """Enhanced fallback with better DuckDB vs web detection"""
        logger.info("🔍 Using enhanced fallback pattern detection...")
        
        question_lower = question.lower()
        
        # Extract S3 paths (more precise)
        s3_patterns = [
            r's3://[a-zA-Z0-9.\-_]+/[^\s\)\'"]+\.(?:parquet|csv)',
            r's3://[a-zA-Z0-9.\-_]+/[^\s\)\'"]*metadata\.parquet',
        ]
        
        extracted_s3_paths = []
        for pattern in s3_patterns:
            try:
                matches = re.findall(pattern, question, re.IGNORECASE)
                extracted_s3_paths.extend(matches)
            except re.error:
                continue
        
        # Extract SQL queries
        sql_patterns = [
            r'(?:SELECT|INSTALL|LOAD)\s[^;]+(?:FROM|httpfs|parquet)[^;]*(?:;|$)',
            r'read_parquet\s*\([^\)]+\)',
        ]
        
        extracted_sql = []
        for pattern in sql_patterns:
            try:
                matches = re.findall(pattern, question, re.IGNORECASE | re.DOTALL)
                extracted_sql.extend([m.strip() for m in matches])
            except re.error:
                continue
        
        # Extract web URLs (excluding citations)
        # Look for actual scrape targets, not citations
        web_url_patterns = [
            r'(?:scrape|fetch|crawl|extract.*from)\s+(?:https?://[^\s\)\]]+)',
            r'https?://(?:wikipedia|github|stackoverflow|reddit|news)[^\s\)\]]+',
        ]
        
        extracted_web_urls = []
        for pattern in web_url_patterns:
            try:
                matches = re.findall(pattern, question, re.IGNORECASE)
                extracted_web_urls.extend(matches)
            except re.error:
                continue
        
        # Find citation links (not for scraping)
        citation_patterns = [
            r'\[([^\]]+)\]\s*\(\s*(https?://[^\s\)]+)\s*\)',  # Markdown links
            r'downloaded from \[([^\]]+)\]\s*\(\s*(https?://[^\s\)]+)\s*\)',  # Context citations
        ]
        
        citation_links = []
        for pattern in citation_patterns:
            try:
                matches = re.findall(pattern, question)
                citation_links.extend([match[1] if isinstance(match, tuple) else match for match in matches])
            except re.error:
                continue
        
        # Detect needs using safe pattern matching
        def safe_pattern_match(patterns, text):
            for pattern in patterns:
                try:
                    if re.search(pattern, text, re.IGNORECASE):
                        return True
                except re.error:
                    continue
            return False
        
        needs_duckdb = (
            bool(extracted_s3_paths) or
            bool(extracted_sql) or
            safe_pattern_match(self.duckdb_patterns, question_lower)
        )
        
        needs_web = (
            bool(extracted_web_urls) and
            not needs_duckdb  # Don't web scrape if we have DuckDB data
        )
        
        needs_csv = (
            bool(kwargs.get('csv_files')) or
            safe_pattern_match(self.csv_patterns, question_lower)
        ) and not needs_duckdb  # DuckDB takes priority
        
        needs_viz = safe_pattern_match(self.viz_patterns, question_lower)
        
        needs_stats = any(keyword in question_lower for keyword in [
            'regression', 'correlation', 'statistical', 'slope', 'analysis'
        ])
        
        # Determine primary data source
        if needs_duckdb:
            primary_source = "cloud_duckdb"
        elif needs_web:
            primary_source = "web_scraping"
        elif needs_csv:
            primary_source = "csv_analysis"
        else:
            primary_source = "mixed"
        
        # Determine data scale
        if any(term in question_lower for term in ['16m', '1tb', 'massive', 'millions']):
            data_scale = "massive"
        elif any(term in question_lower for term in ['large', 'big data', 'gb']):
            data_scale = "large"
        else:
            data_scale = "medium"
        
        # Complexity assessment
        complexity_score = (
            len(re.findall(r'\?', question)) +
            len(extracted_s3_paths) +
            len(extracted_sql) +
            (2 if needs_stats else 0) +
            (1 if needs_viz else 0)
        )
        
        if complexity_score > 4:
            complexity = "complex"
        elif complexity_score > 2:
            complexity = "medium"
        else:
            complexity = "simple"
        
        # Expected output
        if needs_viz and 'json' in question_lower:
            expected_output = "mixed"
        elif needs_viz:
            expected_output = "image"
        elif 'json' in question_lower:
            expected_output = "json"
        else:
            expected_output = "text"
        
        fallback_result = {
            "primary_data_source": primary_source,
            "extracted_s3_paths": extracted_s3_paths,
            "extracted_sql_queries": extracted_sql,
            "extracted_web_urls": extracted_web_urls,
            "extracted_local_files": [],
            "needs_duckdb": needs_duckdb,
            "needs_web_scraping": needs_web,
            "needs_csv_analysis": needs_csv,
            "needs_visualization": needs_viz,
            "needs_statistical_analysis": needs_stats,
            "data_scale": data_scale,
            "analysis_complexity": complexity,
            "expected_output": expected_output,
            "confidence": 0.8,
            "reasoning": "Enhanced pattern-based detection with DuckDB priority",
            "key_technologies": self._determine_technologies(needs_duckdb, needs_web, needs_csv, needs_viz),
            "workflow_steps": self._determine_workflow_steps(question, needs_duckdb, needs_web, needs_csv, needs_viz),
            "cloud_config": {"region": "ap-south-1", "requires_credentials": False},
            "citation_links_found": citation_links,
            "actual_scrape_targets": extracted_web_urls,
            "fallback_used": True
        }
        
        logger.info(f"📋 Fallback detected: {primary_source}, {complexity} complexity")
        logger.info(f"☁️ S3 paths: {len(extracted_s3_paths)}, 🔧 SQL: {len(extracted_sql)}")
        logger.info(f"🌐 Web targets: {len(extracted_web_urls)}, 📄 Citations: {len(citation_links)}")
        
        return fallback_result
    
    def _determine_technologies(self, needs_duckdb, needs_web, needs_csv, needs_viz):
        """Determine required technologies"""
        technologies = ["pandas", "json"]
        
        if needs_duckdb:
            technologies.extend(["duckdb"])
        if needs_web:
            technologies.extend(["httpx", "beautifulsoup4"])
        if needs_csv:
            technologies.extend(["numpy"])
        if needs_viz:
            technologies.extend(["matplotlib", "seaborn", "base64", "io"])
            
        return list(set(technologies))
    
    def _determine_workflow_steps(self, question, needs_duckdb, needs_web, needs_csv, needs_viz):
        """Determine workflow steps"""
        steps = []
        
        if needs_duckdb:
            steps.append("Setup DuckDB with S3 extensions")
            steps.append("Execute SQL queries on cloud data")
        if needs_web:
            steps.append("Scrape web data")
        if needs_csv:
            steps.append("Load local CSV data")
        if "regression" in question.lower():
            steps.append("Perform regression analysis")
        if needs_viz:
            steps.append("Create visualizations")
        if "json" in question.lower():
            steps.append("Format output as JSON")
            
        return steps
    
    def get_web_context(self, url: str, sample_size: int = 15000) -> Optional[Dict[str, Any]]:
        """Enhanced web scraping - only used when actually needed"""
        if not url:
            return None
            
        logger.info(f"🌐 Fetching web context from: {url}")
        
        # Clean URL (remove markdown artifacts)
        url = url.strip('().,')
        if not url.startswith(('http://', 'https://')):
            logger.error(f"❌ Invalid URL format: {url}")
            return None
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }
        
        try:
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                time.sleep(1)  # Be respectful
                
                response = client.get(url, headers=headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                context = self._extract_web_context(soup, url, sample_size)
                
                logger.info(f"✅ Web context extracted ({len(context.get('content', ''))} chars)")
                return context
                
        except Exception as e:
            logger.error(f"❌ Web fetch error: {e}")
            return None
    
    def _extract_web_context(self, soup: BeautifulSoup, url: str, sample_size: int) -> Dict[str, Any]:
        """Extract comprehensive context from web page"""
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title"
        
        # Extract tables
        tables = []
        for i, table in enumerate(soup.find_all('table')[:5]):
            table_data = self._extract_table_data(table)
            if table_data:
                tables.append({
                    'caption': f"Table {i+1}",
                    'headers': table_data.get('headers', []),
                    'rows': table_data.get('rows', [])[:20],
                    'row_count': len(table_data.get('rows', []))
                })
        
        # Get main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        all_text = main_content.get_text() if main_content else soup.get_text()
        all_text = re.sub(r'\s+', ' ', all_text).strip()
        
        return {
            'content': all_text[:sample_size],
            'title': title_text,
            'url': url,
            'tables': tables,
            'table_count': len(tables)
        }
    
    def _extract_table_data(self, table) -> Optional[Dict[str, Any]]:
        """Extract structured data from HTML table"""
        try:
            rows = table.find_all('tr')
            if not rows:
                return None
            
            headers = []
            header_row = rows[0]
            header_cells = header_row.find_all(['th', 'td'])
            headers = [cell.get_text().strip() for cell in header_cells]
            
            data_rows = []
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = [cell.get_text().strip() for cell in cells]
                    if any(cell for cell in row_data):
                        data_rows.append(row_data)
            
            return {'headers': headers, 'rows': data_rows}
            
        except Exception:
            return None
    
    def generate_analysis_code(self, question: str, context_info: Dict[str, Any], 
                             analysis_requirements: Dict[str, Any], error_context: Optional[str] = None) -> Optional[str]:
        """Enhanced code generation with DuckDB focus"""
        logger.info("🤖 Generating analysis code...")
        
        primary_source = analysis_requirements.get('primary_data_source', 'unknown')
        
        if primary_source == 'cloud_duckdb':
            system_prompt = self._get_duckdb_system_prompt()
        else:
            system_prompt = self._get_general_system_prompt()
        
        context_str = self._build_context_string(question, context_info, analysis_requirements, error_context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_str}
                ],
                temperature=0.1,
                timeout=30
            )
            
            code = self._clean_generated_code(response.choices[0].message.content)
            logger.info("✅ Code generated successfully")
            return code
            
        except Exception as e:
            logger.error(f"❌ Code generation error: {e}")
            return None
    
    def _get_duckdb_system_prompt(self) -> str:
        """System prompt optimized for DuckDB cloud analysis"""
        return """You are an expert Python DuckDB analyst. Generate robust code for cloud data analysis.

CRITICAL REQUIREMENTS FOR DUCKDB:
- Import duckdb and set up connection: conn = duckdb.connect()
- Install and load required extensions: conn.execute("INSTALL httpfs; LOAD httpfs;")
- Use read_parquet() for S3 data with proper s3_region parameter
- Execute SQL queries with conn.execute() and fetch results with fetchall() or fetchdf()
- Convert DuckDB results to pandas DataFrames for further analysis
- Handle large datasets efficiently with LIMIT clauses when needed

DUCKDB S3 ACCESS:
- No AWS credentials needed for public buckets
- Use s3_region parameter: read_parquet('s3://bucket/path/*.parquet?s3_region=region')
- Handle partitioned data with wildcard patterns: year=*/court=*/bench=*/*
- Use appropriate SQL aggregations and filtering in the query

DUCKDB SQL RULES:
- Target engine: DuckDB only. Do not invent or borrow functions from Postgres/MySQL/BigQuery
- Date parsing: use strptime() or try_strptime() with correct format strings (e.g. '%d-%m-%Y')
- Date math: use datediff('day', start, end) or (CAST(end AS DATE) - CAST(start AS DATE))
- NULL checks: use IS NULL / IS NOT NULL only (never "IS NOT")
- Regression: use regr_slope(y,x) and regr_intercept(y,x) for linear regression

- Validate SQL with EXPLAIN before execution; if EXPLAIN fails, fix query instead of inventing functions

STATISTICAL ANALYSIS:
- Use pandas/numpy for regression, correlation analysis
- For regression: use scipy.stats.linregress() or sklearn.linear_model.LinearRegression()
- Calculate slopes, R-squared values, and statistical significance

VISUALIZATION:
- Use matplotlib/seaborn for plots
- Convert to base64 PNG: use io.BytesIO(), plt.savefig(), base64.b64encode()
- Keep images under 100KB: use dpi=72, bbox_inches='tight'
- Always plt.close() after saving to prevent memory issues

JSON OUTPUT:
- Return results as JSON using json.dumps() with ensure_ascii=False and indent=2
- Convert all numpy arrays/pandas series to lists with .tolist()
- Handle datetime objects by converting to strings
- Structure output as requested in the question

ERROR HANDLING:
- Wrap DuckDB operations in try-except blocks
- Handle empty query results gracefully
- Validate data before statistical operations
- Check for null/missing values

RETURN ONLY EXECUTABLE PYTHON CODE - no explanations or markdown."""
    
    def _get_general_system_prompt(self) -> str:
        """General system prompt for non-DuckDB analysis"""
        return """You are an expert Python data analyst code generator. Generate robust, production-ready Python code.

CRITICAL REQUIREMENTS:
- Return ONLY executable Python code, no explanations, comments, or markdown
- Handle ALL data type conversions for JSON serialization (use .tolist() for numpy arrays, str() for non-serializable types)
- Use appropriate libraries: httpx+BeautifulSoup for web, pandas+numpy for data, matplotlib for visualizations, duckdb for database queries
- For visualizations: convert to base64 PNG, keep under 100KB, use plt.tight_layout() and dpi=72
- Always return results using json.dumps() with ensure_ascii=False and indent=2
- Include comprehensive error handling and data validation
- Handle missing data, empty datasets, malformed inputs gracefully
- Use try-except blocks around all external operations
- For web scraping: handle timeouts, encoding issues, missing elements
- For CSV: handle different encodings, missing columns, data type issues

COMMON PITFALLS TO AVOID:
- Don't use numpy/pandas objects directly in JSON - convert to Python native types
- Always check if dataframes/lists are empty before processing
- Handle matplotlib figures properly (plt.close() after saving)
- Use proper string handling for text processing
- Validate URLs and file paths before using

"""
    
    def _build_context_string(self, question: str, context_info: Dict[str, Any], 
                             analysis_requirements: Dict[str, Any], error_context: Optional[str]) -> str:
        """Build context string for code generation"""
        context_parts = [f"QUESTION: {question}\n"]
        
        primary_source = analysis_requirements.get('primary_data_source')
        context_parts.append(f"PRIMARY DATA SOURCE: {primary_source}\n")
        
        # Add S3/DuckDB specific context
        if analysis_requirements.get('extracted_s3_paths'):
            context_parts.append("S3 PARQUET PATHS:")
            for path in analysis_requirements['extracted_s3_paths']:
                context_parts.append(f"  {path}")
            context_parts.append("")
        
        if analysis_requirements.get('extracted_sql_queries'):
            context_parts.append("SQL QUERIES TO EXECUTE:")
            for query in analysis_requirements['extracted_sql_queries']:
                context_parts.append(f"  {query}")
            context_parts.append("")
        
        # Add web context if available
        if context_info.get('web_content'):
            web_data = context_info['web_content']
            context_parts.append(f"WEB CONTENT:\n{str(web_data)[:2000]}\n")
        
        # Add cloud config
        if analysis_requirements.get('cloud_config'):
            config = analysis_requirements['cloud_config']
            context_parts.append(f"CLOUD CONFIG: Region={config.get('region')}, Credentials={config.get('requires_credentials')}\n")
        
        context_parts.append(f"ANALYSIS REQUIREMENTS:\n{json.dumps(analysis_requirements, indent=2, default=str)}\n")
        
        if error_context:
            context_parts.append(f"PREVIOUS ERROR TO FIX:\n{error_context}\n")
        
        context_parts.append("""
SPECIFIC INSTRUCTIONS:
1. Focus on the primary data source identified above
2. For DuckDB: Install extensions, execute SQL, convert to pandas, perform analysis
3. For regression: Calculate slopes and create visualizations as requested
4. Return complete results in the exact JSON format specified in the question
5. Handle all data type conversions properly for JSON serialization""")
        
        return "\n".join(context_parts)
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code"""
        code = code.strip()
        if code.startswith('```python'):
            code = code[9:]
        elif code.startswith('```'):
            code = code[3:]
        if code.endswith('```'):
            code = code[:-3]
        return code.strip()
    
    def execute_code_safely(self, code: str, csv_files: Optional[List[str]] = None, 
                          sample_data: Optional[Dict[str, str]] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """Execute code with enhanced monitoring for DuckDB"""
        logger.info("🚀 Executing generated code...")
        
        with tempfile.TemporaryDirectory() as td:
            self._setup_execution_environment(td, csv_files, sample_data)
            
            code_path = os.path.join(td, "generated_analysis.py")
            try:
                with open(code_path, "w", encoding="utf-8") as f:
                    f.write(code)
                
                # Extended timeout for DuckDB operations
                proc = subprocess.run(
                    [sys.executable, code_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=td,
                    env={**os.environ, 'PYTHONPATH': td}
                )
                
                success = proc.returncode == 0
                output = proc.stdout.strip() if success else None
                error = proc.stderr if not success else None
                
                if success:
                    logger.info("✅ Code execution successful")
                else:
                    logger.error(f"❌ Code execution failed: {error}")
                
                return success, output, error
                
            except subprocess.TimeoutExpired:
                logger.error(f"❌ Code execution timeout ({self.timeout}s)")
                return False, None, f"Code execution timeout ({self.timeout}s)"
            except Exception as e:
                logger.error(f"❌ Execution error: {e}")
                return False, None, str(e)
    
    def _setup_execution_environment(self, temp_dir: str, csv_files: Optional[List[str]], 
                                   sample_data: Optional[Dict[str, str]]) -> None:
        """Setup execution environment"""
        if csv_files:
            for csv_file in csv_files:
                if os.path.exists(csv_file):
                    try:
                        shutil.copy2(csv_file, os.path.join(temp_dir, os.path.basename(csv_file)))
                        logger.info(f"✅ Copied {csv_file}")
                    except Exception as e:
                        logger.error(f"❌ Failed to copy {csv_file}: {e}")
        
        if sample_data:
            for filename, content in sample_data.items():
                try:
                    filepath = os.path.join(temp_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"✅ Created sample file: {filename}")
                except Exception as e:
                    logger.error(f"❌ Error creating {filename}: {e}")
    
    def fix_code_with_llm(self, question: str,original_code: str, error: str, context_info: Dict[str, Any]) -> Optional[str]:
        """Enhanced code fixing with DuckDB-specific error handling"""
        logger.info("🔧 LLM fixing code...")
        
        fix_prompt = f"""Fix this Python code that failed with an error. Check the code and the Question 
        Then fix the code According to question

Question:
{question}

ORIGINAL CODE:
{original_code}

ERROR MESSAGE:
{error}

CONTEXT INFO:
{json.dumps(context_info, indent=2, default=str)}

COMMON DUCKDB ERROR FIXES:
- Missing extensions: Add conn.execute("INSTALL httpfs; LOAD httpfs;")
- S3 region issues: Ensure s3_region parameter is included in read_parquet()
- Connection issues: Use conn = duckdb.connect() and proper conn.execute()
- Result handling: Use fetchall(), fetchdf(), or df() to get results
- Memory issues: Add LIMIT clauses for large datasets
- JSON serialization: Convert numpy arrays with .tolist(), handle datetime objects

GENERAL ERROR FIXES:
- Import errors: Add missing imports (import json, import base64, etc.)
- Data type errors: Handle None values, convert types properly
- File access: Use correct file paths, handle encoding issues
- Matplotlib: Use plt.tight_layout(), plt.close(), proper base64 encoding

REQUIREMENTS:
- Return ONLY the corrected Python code
- Fix ALL issues identified in the error
- Maintain the original functionality
- Add proper error handling where missing
- For DuckDB: ensure proper connection setup and query execution"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": fix_prompt}],
                temperature=0.1,
                timeout=20
            )
            
            fixed_code = self._clean_generated_code(response.choices[0].message.content)
            logger.info("🔧 Code fix generated")
            return fixed_code
            
        except Exception as e:
            logger.error(f"❌ Code fix error: {e}")
            return None
    
    def analyze(self, question: str, **kwargs) -> Dict[str, Any]:
        """Enhanced main analysis method with DuckDB priority"""
        logger.info("🎯 Starting Enhanced LLM-Driven Analysis")
        logger.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        try:
            # Step 1: Analyze requirements with enhanced detection
            analysis_requirements = self.detect_and_prepare_context(question, **kwargs)
            
            # Step 2: Gather context (skip web scraping for DuckDB-focused queries)
            context_info = self._gather_context(analysis_requirements, **kwargs)
            
            # Step 3: Generate and execute code with retries
            return self._execute_with_retries(question, context_info, analysis_requirements, **kwargs)
            
        except Exception as e:
            logger.error(f"❌ Critical analysis error: {e}")
            return {
                "success": False,
                "error": f"Critical analysis failure: {str(e)}",
                "fallback_available": True
            }
    
    def _gather_context(self, analysis_requirements: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Gather context - prioritize DuckDB, skip unnecessary web scraping"""
        context_info = {}
        
        # Only scrape web if it's the primary data source and we have valid targets
        if (analysis_requirements.get('primary_data_source') == 'web_scraping' and
            analysis_requirements.get('actual_scrape_targets')):
            
            url = analysis_requirements['actual_scrape_targets'][0]
            web_content = self.get_web_context(url, 15000)
            if web_content:
                context_info['web_content'] = web_content
        else:
            logger.info("🚫 Skipping web scraping - not needed for this analysis type")
        
        # Handle CSV data if needed
        if analysis_requirements.get('needs_csv_analysis') and kwargs.get('csv_files'):
            csv_info = {}
            for csv_file in kwargs['csv_files']:
                if os.path.exists(csv_file):
                    try:
                        df_sample = pd.read_csv(csv_file, nrows=5)
                        csv_info[csv_file] = {
                            'columns': df_sample.columns.tolist(),
                            'dtypes': df_sample.dtypes.astype(str).to_dict(),
                            'sample_rows': df_sample.to_dict('records')
                        }
                    except Exception as e:
                        csv_info[csv_file] = {'error': str(e)}
            context_info['csv_info'] = csv_info
        
        if kwargs.get('sample_data'):
            context_info['sample_files'] = list(kwargs['sample_data'].keys())
        
        return context_info
    
    def _execute_with_retries(self, question: str, context_info: Dict[str, Any], 
                            analysis_requirements: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute analysis with intelligent retries"""
        code = self.generate_analysis_code(question, context_info, analysis_requirements)
        if not code:
            return {"success": False, "error": "LLM could not generate analysis code"}
        
        original_code = code
        
        for attempt in range(self.max_retries):
            logger.info(f"🔄 Execution attempt {attempt + 1}/{self.max_retries}")
            
            success, output, error = self.execute_code_safely(
                code, 
                kwargs.get('csv_files'),
                kwargs.get('sample_data')
            )
            
            if success and output:
                return self._process_successful_result(output, code, attempt, analysis_requirements)
            
            # Handle failure
            logger.error(f"❌ Attempt {attempt + 1} failed: {error}")
            
            if attempt < self.max_retries - 1:
                fixed_code = self.fix_code_with_llm(code, error, context_info)
                if fixed_code and fixed_code != code:
                    code = fixed_code
                    logger.info("🔧 Applied LLM code fix")
                else:
                    logger.warning("❌ LLM could not provide a fix")
        
        return {
            "success": False,
            "error": f"Analysis failed after {self.max_retries} attempts",
            "final_error": error,
            "original_code": original_code,
            "final_code": code,
            "analysis_requirements": analysis_requirements
        }
    
    def _process_successful_result(self, output: str, code: str, attempts: int, 
                                 analysis_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Process successful analysis result"""
        logger.info("🎉 Analysis completed successfully!")
        
        # Try parsing as JSON
        try:
            result_data = json.loads(output)
            return {
                "success": True,
                "result": result_data,
                "generated_code": code,
                "attempts": attempts + 1,
                "primary_data_source": analysis_requirements.get('primary_data_source', 'unknown'),
                "data_scale": analysis_requirements.get('data_scale', 'unknown'),
                "complexity": analysis_requirements.get('analysis_complexity', 'unknown'),
                "output_format": "json",
                "analysis_requirements": analysis_requirements
            }
        except json.JSONDecodeError:
            return {
                "success": True,
                "result": output,
                "generated_code": code,
                "attempts": attempts + 1,
                "primary_data_source": analysis_requirements.get('primary_data_source', 'unknown'),
                "data_scale": analysis_requirements.get('data_scale', 'unknown'),
                "complexity": analysis_requirements.get('analysis_complexity', 'unknown'),
                "output_format": "text",
                "note": "Result not in JSON format",
                "analysis_requirements": analysis_requirements
            }

# =========================
# API Endpoint
# =========================
# @app.post("/api/")
# async def analyze_endpoint(
#     question_file: UploadFile = File(..., description="Text file with the question"),
#     image_file: Optional[UploadFile] = File(None, description="Optional image file"),
#     data_file: Optional[UploadFile] = File(None, description="Optional CSV file"),
# ):
#     tmpdir = tempfile.mkdtemp()
#     try:
#         # Save question
#         q_path = os.path.join(tmpdir, question_file.filename)
#         with open(q_path, "wb") as f:
#             f.write(await question_file.read())
#         with open(q_path, "r", encoding="utf-8") as f:
#             question_text = f.read()

#         csv_files = []
#         if data_file:
#             csv_path = os.path.join(tmpdir, data_file.filename)
#             with open(csv_path, "wb") as f:
#                 f.write(await data_file.read())
#             csv_files.append(csv_path)

#         if image_file:
#             img_path = os.path.join(tmpdir, image_file.filename)
#             with open(img_path, "wb") as f:
#                 f.write(await image_file.read())
#             logger.info(f"📥 Received image: {img_path}")

#         # Run analysis
#         analyst = PureLLMAnalyst()
#         result = analyst.analyze(question_text, csv_files=csv_files)
#         print(result['result'])
#         return result['result']

#     except Exception as e:
#         logger.error(f"❌ API error: {e}", exc_info=True)
#         return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

#     finally:
#         shutil.rmtree(tmpdir, ignore_errors=True)



@app.post("/api/")
async def analyze_endpoint(
    question_file: UploadFile = File(..., alias="questions.txt"),
    image_files: Optional[List[UploadFile]] = File(None, alias="image.png"),
    data_files: Optional[List[UploadFile]] = File(None, alias="data"),
):
    tmpdir = tempfile.mkdtemp()
    try:
        # Save the question file
        q_path = os.path.join(tmpdir, question_file.filename)
        with open(q_path, "wb") as f:
            f.write(await question_file.read())
        with open(q_path, "r", encoding="utf-8") as f:
            question_text = f.read()

        # Save all CSV files
        csv_paths = []
        if data_files:
            for upload in data_files:
                file_path = os.path.join(tmpdir, upload.filename)
                with open(file_path, "wb") as f:
                    f.write(await upload.read())
                csv_paths.append(file_path)

        # Save all image files
        image_paths = []
        if image_files:
            for upload in image_files:
                file_path = os.path.join(tmpdir, upload.filename)
                with open(file_path, "wb") as f:
                    f.write(await upload.read())
                image_paths.append(file_path)
                logger.info(f"📥 Received image: {file_path}")

        # Run analysis (placeholder)
        analyst = PureLLMAnalyst()
        result = analyst.analyze(question_text, csv_files=csv_paths)

        print(result['result'])
        return result['result']

    except Exception as e:
        logger.error(f"❌ API error: {e}", exc_info=True)
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# =========================
# Run with: uvicorn main:app --reload
# =========================
