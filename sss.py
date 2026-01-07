import os
import zipfile
import io

# Conteúdo dos arquivos
files_content = {
    "__main__.py": '''#!/usr/bin/env python3
"""
Entrypoint principal para execução via `python -m barenet`.
"""
import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
''',

    "cli.py": '''#!/usr/bin/env python3
"""
Interface de linha de comando usando Click.
"""
import click
from typing import Optional, List
from .config import Config, load_config
from .engine import Engine
from .logging_config import init_logging
import asyncio

# Contexto compartilhado entre comandos
@click.group()
@click.option("--db-path", default=None, help="Caminho para o banco de dados SQLite")
@click.option("--safe-mode/--no-safe-mode", default=None, help="Modo de segurança ativado")
@click.option("--local-only", is_flag=True, help="Restringir a URLs locais")
@click.option("--log-level", default="INFO", help="Nível de log (DEBUG, INFO, WARNING, ERROR)")
@click.pass_context
def cli(ctx, db_path, safe_mode, local_only, log_level):
    """BareNet - Scanner passivo de aplicações web"""
    # Configurar logging primeiro
    init_logging(log_level)
    
    # Carregar configuração com overrides
    cfg = load_config()
    if db_path:
        cfg.DB_PATH = db_path
    if safe_mode is not None:
        cfg.SAFE_MODE = safe_mode
    if local_only:
        # Atualizar configuração para restringir a hosts locais
        cfg.LOCAL_ONLY = True
    
    # Criar engine e armazenar no contexto
    ctx.obj = Engine(cfg)

@cli.command()
@click.argument("source")
@click.option("--force", is_flag=True, help="Forçar reindexação mesmo se já existir")
@click.option("--concurrency", default=5, help="Número máximo de requisições paralelas")
@click.option("--yes", is_flag=True, help="Confirmar automaticamente operações perigosas")
@click.pass_obj
def index(engine, source, force, concurrency, yes):
    """Indexar URLs de um arquivo ou plugin"""
    import warnings
    from .plugins import localfile
    
    if force and not yes:
        click.confirm("Forçar reindexação pode sobrepor dados existentes. Continuar?", abort=True)
    
    try:
        # Detectar tipo de fonte
        if os.path.isfile(source):
            urls = localfile.iter_urls_from_file(source)
        else:
            # Tentar carregar como plugin
            click.echo(f"Plugin não implementado: {source}", err=True)
            return
        
        # Executar indexação
        asyncio.run(engine.index(urls, concurrency))
        click.echo("Indexação concluída")
    except Exception as e:
        click.echo(f"Erro durante indexação: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.argument("query")
@click.option("--page-size", default=10, help="Resultados por página")
@click.option("--filter", "filters", multiple=True, help="Filtros (ex: tls:none, has_forms)")
@click.option("--min-score", type=float, help="Pontuação mínima")
@click.option("--page", default=1, help="Número da página")
@click.pass_obj
def search(engine, query, page_size, filters, min_score, page):
    """Buscar no índice"""
    import json
    from .searcher import SearchPage
    
    # Parse filters
    filter_dict = {}
    for f in filters:
        if ":" in f:
            k, v = f.split(":", 1)
            filter_dict[k] = v
        else:
            filter_dict[f] = True
    
    if min_score is not None:
        filter_dict["min_score"] = min_score
    
    # Executar busca
    result_page = asyncio.run(engine.search(query, filter_dict, page, page_size))
    
    # Exibir resultados
    for i, result in enumerate(result_page.results, 1):
        click.echo(f"{i}. [{result.get('score', 0):.2f}] {result.get('title', 'Sem título')}")
        click.echo(f"   {result.get('url')}")
        if result.get('snippet'):
            click.echo(f"   {result.get('snippet')[:100]}...")
        click.echo()
    
    click.echo(f"Página {page}/{result_page.total_pages} ({result_page.total} resultados)")

@cli.command()
@click.argument("id_or_url")
@click.option("--renderer", default="w3m", help="Renderizador (w3m, links)")
@click.pass_obj
def inspect(engine, id_or_url, renderer):
    """Inspecionar página específica"""
    try:
        engine.inspect(id_or_url, renderer)
    except Exception as e:
        click.echo(f"Erro ao inspecionar: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.option("--format", "fmt", default="json", help="Formato (json, csv, ndjson)")
@click.option("--output", default=None, help="Arquivo de saída (stdout se não especificado)")
@click.option("--include-html", is_flag=True, help="Incluir HTML no export")
@click.argument("ids", nargs=-1)
@click.pass_obj
def export(engine, fmt, output, include_html, ids):
    """Exportar resultados"""
    from .exporters import export_json, export_csv, export_ndjson
    
    if not ids:
        click.echo("Forneça IDs para exportar", err=True)
        return
    
    # Buscar registros
    records = []
    for id_str in ids:
        try:
            record = engine.get_record(int(id_str))
            if not include_html and "html_blob" in record:
                del record["html_blob"]
            records.append(record)
        except Exception as e:
            click.echo(f"Erro ao buscar ID {id_str}: {e}", err=True)
    
    # Exportar
    try:
        if output:
            out_file = open(output, "w", encoding="utf-8")
        else:
            out_file = click.get_text_stream("stdout")
        
        if fmt == "json":
            import json
            json.dump(records, out_file, indent=2, ensure_ascii=False)
        elif fmt == "csv":
            import csv
            if records:
                writer = csv.DictWriter(out_file, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
        elif fmt == "ndjson":
            import json
            for record in records:
                out_file.write(json.dumps(record, ensure_ascii=False) + "\\n")
        else:
            click.echo(f"Formato não suportado: {fmt}", err=True)
            return
        
        if output:
            out_file.close()
            click.echo(f"Exportado para {output}")
            
    except Exception as e:
        click.echo(f"Erro durante export: {e}", err=True)
        raise click.Abort()

def main():
    """Função principal invocada pelo entrypoint"""
    try:
        cli()
    except Exception as e:
        click.echo(f"Erro fatal: {e}", err=True)
        return 1
    return 0
''',

    "config.py": '''#!/usr/bin/env python3
"""
Configurações centrais do BareNet.
"""
import os
import tomllib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

@dataclass
class Config:
    """Configurações do sistema."""
    
    # Caminhos
    DB_PATH: str = "barenet.db"
    CACHE_DIR: str = os.path.expanduser("~/.barenet/cache")
    CONFIG_DIR: str = os.path.expanduser("~/.barenet")
    
    # HTTP Client
    USER_AGENT: str = "BareNet/0.1 (Passive Security Scanner)"
    REQUEST_TIMEOUT: int = 15
    MAX_REDIRECTS: int = 5
    MAX_HTML_BYTES: int = 2_000_000  # 2MB
    MAX_RESPONSE_BYTES: int = 10_000_000  # 10MB
    
    # Rate limiting
    RATE_LIMIT_PER_HOST: float = 1.0  # requests per second
    MAX_PARALLEL_HOSTS: int = 5
    REQUEST_DELAY: float = 0.1  # delay between requests
    
    # Indexação
    DEFAULT_PAGE_SIZE: int = 10
    CACHE_TTL_DAYS: int = 30
    
    # Segurança
    SAFE_MODE: bool = True
    LOCAL_ONLY: bool = False  # Restringir a hosts locais
    ALLOWED_SCHEMES: tuple = ("http", "https")
    
    # Heurísticas
    SCORE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "fts": 0.6,
        "heuristics": 0.3,
        "recency": 0.1
    })
    
    # Renderização
    MAX_RENDER_BYTES: int = 1_000_000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    def __post_init__(self):
        """Garantir que diretórios existam."""
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.CONFIG_DIR, exist_ok=True)

def load_config() -> Config:
    """
    Carregar configuração de múltiplas fontes:
    1. Valores padrão
    2. Arquivo ~/.barenet/config.toml
    3. Variáveis de ambiente BARENET_*
    """
    cfg = Config()
    
    # 1. Carregar do arquivo de configuração
    config_file = Path(cfg.CONFIG_DIR) / "config.toml"
    if config_file.exists():
        try:
            with open(config_file, "rb") as f:
                file_config = tomllib.load(f)
            
            # Aplicar configurações do arquivo
            for key, value in file_config.items():
                if hasattr(cfg, key.upper()):
                    setattr(cfg, key.upper(), value)
        except Exception as e:
            import warnings
            warnings.warn(f"Erro ao carregar config.toml: {e}")
    
    # 2. Sobrescrever com variáveis de ambiente
    env_prefix = "BARENET_"
    for env_key, env_value in os.environ.items():
        if env_key.startswith(env_prefix):
            config_key = env_key[len(env_prefix):]
            # Converter para tipo apropriado
            if config_key in ["SAFE_MODE", "LOCAL_ONLY"]:
                value = env_value.lower() in ("true", "1", "yes", "on")
            elif config_key in ["REQUEST_TIMEOUT", "MAX_HTML_BYTES", "MAX_PARALLEL_HOSTS"]:
                value = int(env_value)
            elif config_key in ["RATE_LIMIT_PER_HOST", "REQUEST_DELAY"]:
                value = float(env_value)
            else:
                value = env_value
            
            if hasattr(cfg, config_key):
                setattr(cfg, config_key, value)
    
    return cfg

# Instância global
_config_instance: Optional[Config] = None

def get_config() -> Config:
    """Obter instância singleton da configuração."""
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance
''',

    "logging_config.py": '''#!/usr/bin/env python3
"""
Configuração de logging unificada.
"""
import logging
import sys
from typing import Optional
from .config import get_config

class SensitiveDataFilter(logging.Filter):
    """Filtro para remover dados sensíveis dos logs."""
    
    SENSITIVE_HEADERS = {
        "authorization", "cookie", "set-cookie",
        "x-api-key", "x-auth-token", "x-csrf-token"
    }
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Limpar mensagens que podem conter dados sensíveis
        if hasattr(record, "msg"):
            msg = str(record.msg)
            # Remover headers sensíveis de forma simplificada
            for header in self.SENSITIVE_HEADERS:
                if header in msg.lower():
                    record.msg = "[SENSITIVE DATA REDACTED]"
                    break
        return True

def init_logging(
    level: str = "INFO",
    file: Optional[str] = None,
    sensitive_filter: bool = True
) -> None:
    """
    Inicializar configuração de logging.
    
    Args:
        level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file: Arquivo para logs (opcional)
        sensitive_filter: Ativar filtro de dados sensíveis
    """
    # Configurar formato
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configurar handler do console
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    
    # Obter logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remover handlers existentes
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Adicionar console handler
    root_logger.addHandler(console_handler)
    
    # Adicionar filtro de dados sensíveis
    if sensitive_filter:
        console_handler.addFilter(SensitiveDataFilter())
    
    # Configurar arquivo de log se especificado
    if file:
        try:
            file_handler = logging.FileHandler(file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            if sensitive_filter:
                file_handler.addFilter(SensitiveDataFilter())
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.warning(f"Não foi possível configurar log para arquivo {file}: {e}")
    
    # Configurar loggers de bibliotecas externas
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    root_logger.info(f"Logging configurado com nível {level}")

def get_logger(name: str) -> logging.Logger:
    """Obter logger com nome específico."""
    return logging.getLogger(name)
''',

    "engine.py": '''#!/usr/bin/env python3
"""
Orquestrador principal do BareNet.
"""
import asyncio
from typing import Iterator, Optional, Dict, Any, List
from dataclasses import dataclass
from .config import Config
from .httpclient import HttpClient
from .indexer import parse_and_store
from .searcher import search, SearchPage
from .db.storage import Storage
from .cache import Cache
import logging

logger = logging.getLogger(__name__)

@dataclass
class EngineContext:
    """Contexto compartilhado para operações do engine."""
    config: Config
    storage: Storage
    http_client: HttpClient
    cache: Cache

class Engine:
    """Orquestrador principal."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.storage = Storage(self.config.DB_PATH)
        self.http_client = HttpClient(self.config)
        self.cache = Cache(self.config.CACHE_DIR)
        self.ctx = EngineContext(
            config=self.config,
            storage=self.storage,
            http_client=self.http_client,
            cache=self.cache
        )
    
    async def index(self, source: Iterator[str], concurrency: int = 5) -> Dict[str, Any]:
        """
        Indexar URLs de uma fonte.
        
        Args:
            source: Iterador de URLs
            concurrency: Número máximo de requisições paralelas
            
        Returns:
            Estatísticas da indexação
        """
        from collections import defaultdict
        import time
        
        stats = defaultdict(int)
        start_time = time.time()
        
        # Converter iterator para lista para processamento
        urls = list(source)
        stats["total_urls"] = len(urls)
        
        logger.info(f"Iniciando indexação de {len(urls)} URLs com concorrência {concurrency}")
        
        # Semáforo para controlar concorrência
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_url(url: str):
            """Processar uma URL individual."""
            async with semaphore:
                try:
                    # Verificar cache
                    cached = self.cache.get(url)
                    if cached:
                        stats["cached"] += 1
                        fetch_result = cached
                    else:
                        # Fetch HTTP
                        fetch_result = await self.http_client.fetch(url)
                        stats["fetched"] += 1
                        
                        # Armazenar em cache
                        if fetch_result.status == 200:
                            self.cache.put(url, fetch_result)
                    
                    # Indexar
                    page_id = await parse_and_store(url, fetch_result, self.ctx)
                    if page_id:
                        stats["indexed"] += 1
                        logger.debug(f"Indexado: {url} (ID: {page_id})")
                    else:
                        stats["failed"] += 1
                        
                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"Erro ao processar {url}: {e}")
                    # Não propagar exceção para não interromper outras URLs
        
        # Criar e executar tasks
        tasks = [process_url(url) for url in urls]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calcular estatísticas finais
        elapsed = time.time() - start_time
        stats["elapsed_seconds"] = elapsed
        stats["urls_per_second"] = stats["fetched"] / elapsed if elapsed > 0 else 0
        
        logger.info(
            f"Indexação concluída: "
            f"{stats['indexed']} indexadas, "
            f"{stats['cached']} do cache, "
            f"{stats['errors']} erros, "
            f"{stats['failed']} falhas"
        )
        
        return dict(stats)
    
    async def search(
        self,
        query: str,
        filters: Dict[str, Any],
        page: int = 1,
        page_size: int = 10
    ) -> SearchPage:
        """
        Buscar no índice.
        
        Args:
            query: Termos de busca
            filters: Filtros aplicados
            page: Número da página
            page_size: Tamanho da página
            
        Returns:
            Página de resultados
        """
        return await search(query, filters, page, page_size, self.ctx)
    
    def inspect(self, id_or_url: str, renderer: str = "w3m") -> None:
        """
        Inspecionar página específica.
        
        Args:
            id_or_url: ID numérico ou URL
            renderer: Renderizador a usar
        """
        from .inspect import render_in_terminal
        
        # Tentar interpretar como ID
        page_id = None
        if id_or_url.isdigit():
            page_id = int(id_or_url)
            record = self.storage.get_page(page_id)
            if not record:
                raise ValueError(f"Página com ID {page_id} não encontrada")
            html = record.get("html_blob", b"").decode("utf-8", errors="replace")
            url = record.get("url", "")
        else:
            # É uma URL - tentar buscar
            url = id_or_url
            # TODO: Buscar URL se não estiver no banco
            raise NotImplementedError("Busca por URL ainda não implementada")
        
        # Renderizar
        render_in_terminal(html, renderer, url)
    
    def get_record(self, page_id: int) -> Dict[str, Any]:
        """Obter registro completo pelo ID."""
        return self.storage.get_page(page_id)
    
    async def close(self):
        """Liberar recursos."""
        await self.http_client.close()
        self.storage.close()
''',

    "httpclient.py": '''#!/usr/bin/env python3
"""
Cliente HTTP assíncrono com rate limiting e segurança.
"""
import asyncio
import httpx
import ssl
from typing import Optional, Dict, Any, NamedTuple
from dataclasses import dataclass
from urllib.parse import urlparse
import logging
from .config import Config

logger = logging.getLogger(__name__)

class FetchResult(NamedTuple):
    """Resultado de uma requisição HTTP."""
    status: int
    headers: Dict[str, str]
    body: bytes
    final_url: str
    tls_info: Optional[Dict[str, Any]] = None
    truncated: bool = False
    error: Optional[str] = None

@dataclass
class HostLimiter:
    """Limitador de requisições por host."""
    semaphore: asyncio.Semaphore
    last_request: float = 0
    request_count: int = 0

class HttpClient:
    """Cliente HTTP com rate limiting e controle de concorrência."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
        self.host_limiters: Dict[str, HostLimiter] = {}
        self.total_semaphore = asyncio.Semaphore(config.MAX_PARALLEL_HOSTS)
        
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def start(self):
        """Inicializar cliente HTTP."""
        limits = httpx.Limits(
            max_keepalive_connections=5,
            max_connections=10
        )
        
        self.client = httpx.AsyncClient(
            http2=True,
            timeout=self.config.REQUEST_TIMEOUT,
            limits=limits,
            follow_redirects=True,
            max_redirects=self.config.MAX_REDIRECTS,
            headers={
                "User-Agent": self.config.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
            }
        )
    
    async def close(self):
        """Fechar cliente HTTP."""
        if self.client:
            await self.client.aclose()
    
    def _get_host_limiter(self, url: str) -> HostLimiter:
        """Obter limitador para um host."""
        host = urlparse(url).netloc
        if host not in self.host_limiters:
            self.host_limiters[host] = HostLimiter(
                semaphore=asyncio.Semaphore(1)
            )
        return self.host_limiters[host]
    
    async def fetch(self, url: str) -> FetchResult:
        """
        Buscar uma URL com rate limiting e verificação de segurança.
        
        Args:
            url: URL para buscar
            
        Returns:
            FetchResult com dados da resposta
            
        Raises:
            ValueError: URL inválida ou esquema não permitido
        """
        # Validar URL
        parsed = urlparse(url)
        if not parsed.scheme:
            url = "http://" + url
            parsed = urlparse(url)
        
        if parsed.scheme not in self.config.ALLOWED_SCHEMES:
            raise ValueError(f"Esquema não permitido: {parsed.scheme}")
        
        if self.config.LOCAL_ONLY:
            # Verificar se é host local
            host = parsed.hostname
            if not (host == "localhost" or host.startswith("127.") or host.startswith("192.168.")):
                raise ValueError(f"Acesso a host não-local bloqueado: {host}")
        
        # Aplicar rate limiting por host
        host_limiter = self._get_host_limiter(url)
        
        async with self.total_semaphore:
            async with host_limiter.semaphore:
                # Rate limiting por host
                now = asyncio.get_event_loop().time()
                time_since_last = now - host_limiter.last_request
                if time_since_last < 1.0 / self.config.RATE_LIMIT_PER_HOST:
                    await asyncio.sleep(1.0 / self.config.RATE_LIMIT_PER_HOST - time_since_last)
                
                # Fazer a requisição
                try:
                    return await self._do_fetch(url)
                finally:
                    host_limiter.last_request = asyncio.get_event_loop().time()
                    host_limiter.request_count += 1
    
    async def _do_fetch(self, url: str) -> FetchResult:
        """Executar requisição HTTP real."""
        if not self.client:
            await self.start()
        
        try:
            # Primeiro, HEAD para verificar tamanho e tipo
            head_response = await self.client.head(
                url,
                follow_redirects=True
            )
            
            # Verificar Content-Type
            content_type = head_response.headers.get("content-type", "").lower()
            if not any(ct in content_type for ct in ["text/html", "text/plain", "application/xhtml+xml"]):
                logger.warning(f"Content-Type não-HTML: {content_type} para {url}")
                # Continuar de qualquer forma, mas marcar
            
            # Verificar tamanho do conteúdo
            content_length = head_response.headers.get("content-length")
            if content_length:
                size = int(content_length)
                if size > self.config.MAX_RESPONSE_BYTES:
                    return FetchResult(
                        status=head_response.status_code,
                        headers=dict(head_response.headers),
                        body=b"",
                        final_url=str(head_response.url),
                        truncated=True,
                        error=f"Content too large: {size} bytes"
                    )
            
            # Agora GET com limite de tamanho
            async with self.client.stream("GET", url) as response:
                # Ler corpo com limite
                body = bytearray()
                truncated = False
                
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > self.config.MAX_HTML_BYTES:
                        truncated = True
                        # Adicionar apenas o que falta para atingir o limite
                        remaining = self.config.MAX_HTML_BYTES - len(body)
                        if remaining > 0:
                            body.extend(chunk[:remaining])
                        break
                    body.extend(chunk)
                
                # Coletar informações TLS se disponíveis
                tls_info = None
                if response.http_version == "HTTP/2" or response.http_version == "HTTP/1.1":
                    transport = response.extensions.get("transport")
                    if transport and hasattr(transport, "get_extra_info"):
                        ssl_object = transport.get_extra_info("ssl_object")
                        if ssl_object:
                            tls_info = {
                                "version": ssl_object.version(),
                                "cipher": ssl_object.cipher(),
                                "cert": {
                                    "subject": dict(ssl_object.getpeercert().get("subject", [])),
                                    "issuer": dict(ssl_object.getpeercert().get("issuer", [])),
                                    "notAfter": ssl_object.getpeercert().get("notAfter"),
                                    "notBefore": ssl_object.getpeercert().get("notBefore"),
                                } if ssl_object.getpeercert() else None
                            }
                
                return FetchResult(
                    status=response.status_code,
                    headers=dict(response.headers),
                    body=bytes(body),
                    final_url=str(response.url),
                    tls_info=tls_info,
                    truncated=truncated
                )
                
        except httpx.TimeoutException:
            logger.warning(f"Timeout ao buscar {url}")
            return FetchResult(
                status=0,
                headers={},
                body=b"",
                final_url=url,
                error="Timeout"
            )
        except httpx.RequestError as e:
            logger.warning(f"Erro de requisição para {url}: {e}")
            return FetchResult(
                status=0,
                headers={},
                body=b"",
                final_url=url,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Erro inesperado ao buscar {url}: {e}")
            return FetchResult(
                status=0,
                headers={},
                body=b"",
                final_url=url,
                error=str(e)
            )
''',

    "indexer.py": '''#!/usr/bin/env python3
"""
Processador principal para indexação de páginas.
"""
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from .parser.html_parser import parse_html
from .heuristics import analyze
from .utils import normalize_url, truncate_html
from .engine import EngineContext

logger = logging.getLogger(__name__)

async def parse_and_store(url: str, fetch_result, ctx: EngineContext) -> Optional[int]:
    """
    Processar e armazenar resultado de fetch.
    
    Args:
        url: URL original
        fetch_result: Resultado do fetch
        ctx: Contexto do engine
        
    Returns:
        ID da página armazenada ou None em caso de falha
    """
    try:
        # Decodificar HTML
        html = fetch_result.body.decode("utf-8", errors="replace")
        
        # Parsear HTML
        parsed = parse_html(html, url)
        
        # Aplicar heurísticas
        heuristics_result = analyze(parsed, fetch_result.headers)
        
        # Construir registro
        record = build_record(
            url=url,
            parsed=parsed,
            fetch_result=fetch_result,
            heuristics=heuristics_result
        )
        
        # Armazenar no banco
        page_id = ctx.storage.upsert_page(record)
        
        logger.debug(f"Página indexada: {url} -> ID {page_id}")
        return page_id
        
    except Exception as e:
        logger.error(f"Erro ao processar {url}: {e}")
        
        # Armazenar registro mínimo em caso de erro de parsing
        try:
            minimal_record = {
                "url": url,
                "normalized_url": normalize_url(url),
                "title": "Erro de parsing",
                "snippet": f"Erro: {str(e)[:100]}",
                "html_blob": fetch_result.body[:1000],  # Primeiros 1000 bytes
                "headers_json": json.dumps(dict(fetch_result.headers)),
                "tls_json": json.dumps(fetch_result.tls_info) if fetch_result.tls_info else "{}",
                "heuristics_json": json.dumps({"tags": ["parse_error"], "score": 0.0}),
                "score": 0.0,
                "discovered_at": datetime.utcnow().isoformat() + "Z"
            }
            return ctx.storage.upsert_page(minimal_record)
        except Exception as store_error:
            logger.error(f"Erro ao armazenar registro mínimo: {store_error}")
            return None

def build_record(
    url: str,
    parsed: Dict[str, Any],
    fetch_result,
    heuristics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Construir registro normalizado para armazenamento.
    
    Args:
        url: URL original
        parsed: Resultado do parser HTML
        fetch_result: Resultado do fetch
        heuristics: Resultado das heurísticas
        
    Returns:
        Dicionário com registro normalizado
    """
    from .utils import iso_now
    
    # Normalizar URL
    normalized = normalize_url(url)
    
    # Preparar snippet (primeiro texto útil)
    snippet = ""
    if parsed.get("first_text_blocks"):
        snippet = " ".join(parsed["first_text_blocks"][:2])[:200]
    elif parsed.get("meta_description"):
        snippet = parsed["meta_description"][:200]
    
    # Truncar HTML se necessário
    html_blob = fetch_result.body
    if len(html_blob) > 100000:  # 100KB para armazenamento
        html_blob = html_blob[:100000]
    
    return {
        "url": url,
        "normalized_url": normalized,
        "title": parsed.get("title", "")[:500],
        "snippet": snippet,
        "html_blob": html_blob,
        "headers_json": json.dumps(dict(fetch_result.headers)),
        "tls_json": json.dumps(fetch_result.tls_info) if fetch_result.tls_info else "{}",
        "heuristics_json": json.dumps(heuristics),
        "score": heuristics.get("score", 0.0),
        "discovered_at": iso_now()
    }
''',

    "parser/html_parser.py": '''#!/usr/bin/env python3
"""
Parser HTML para extração de informações estruturais.
"""
from typing import Dict, List, Optional, Any, NamedTuple
from urllib.parse import urljoin, urlparse
import re
from bs4 import BeautifulSoup, Comment

class LinkStruct(NamedTuple):
    """Estrutura para links extraídos."""
    href: str
    text: str
    absolute_url: str
    rel: Optional[str] = None
    target: Optional[str] = None

class FormStruct(NamedTuple):
    """Estrutura para formulários extraídos."""
    method: str
    action: str
    absolute_action: str
    inputs: List[Dict[str, str]]
    has_csrf_token: bool = False

def parse_html(html: str, base_url: str) -> Dict[str, Any]:
    """
    Extrair informações estruturadas de HTML.
    
    Args:
        html: Conteúdo HTML
        base_url: URL base para resolução de links relativos
        
    Returns:
        Dicionário com informações extraídas
    """
    soup = BeautifulSoup(html, 'lxml')
    
    # Título
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    
    # Meta description
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag["content"].strip()
    
    # Headings H1
    h1s = [h1.get_text().strip() for h1 in soup.find_all("h1")]
    
    # Links
    links = extract_links(soup, base_url)
    
    # Formulários
    forms = extract_forms(soup, base_url)
    
    # Primeiros blocos de texto (parágrafos)
    first_text_blocks = []
    for p in soup.find_all("p")[:5]:
        text = p.get_text().strip()
        if text and len(text) > 20:
            first_text_blocks.append(text)
    
    # Comentários HTML
    comments = []
    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment_text = comment.strip()
        if comment_text:
            comments.append(comment_text)
    
    return {
        "title": title,
        "meta_description": meta_desc,
        "h1s": h1s,
        "links": links,
        "forms": forms,
        "first_text_blocks": first_text_blocks,
        "comments": comments
    }

def extract_links(soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
    """Extrair links do HTML."""
    links = []
    
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        
        text = a.get_text().strip()[:200]
        rel = a.get("rel", [""])[0] if a.get("rel") else None
        target = a.get("target")
        
        # Resolver URL absoluta
        try:
            absolute_url = urljoin(base_url, href)
        except:
            absolute_url = href
        
        links.append(LinkStruct(
            href=href,
            text=text,
            absolute_url=absolute_url,
            rel=rel,
            target=target
        )._asdict())
    
    return links

def extract_forms(soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
    """Extrair formulários do HTML."""
    forms = []
    csrf_patterns = [
        r"csrf", r"token", r"authenticity", r"nonce",
        r"requestverification", r"xsrf"
    ]
    
    for form in soup.find_all("form"):
        method = form.get("method", "GET").upper()
        action = form.get("action", "").strip()
        
        # Resolver ação absoluta
        try:
            absolute_action = urljoin(base_url, action)
        except:
            absolute_action = action
        
        # Extrair inputs
        inputs = []
        has_csrf = False
        
        for input_tag in form.find_all("input"):
            input_type = input_tag.get("type", "text").lower()
            input_name = input_tag.get("name", "")
            input_value = input_tag.get("value", "")
            
            inputs.append({
                "type": input_type,
                "name": input_name,
                "value": input_value
            })
            
            # Verificar se é token CSRF
            if input_type == "hidden":
                name_lower = input_name.lower()
                value_lower = input_value.lower()
                for pattern in csrf_patterns:
                    if (re.search(pattern, name_lower) or 
                        (len(input_value) > 20 and re.search(pattern, value_lower))):
                        has_csrf = True
        
        forms.append(FormStruct(
            method=method,
            action=action,
            absolute_action=absolute_action,
            inputs=inputs,
            has_csrf_token=has_csrf
        )._asdict())
    
    return forms
''',

    "db/storage.py": '''#!/usr/bin/env python3
"""
Armazenamento SQLite com FTS5 para busca textual.
"""
import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class Storage:
    """Gerenciador de armazenamento SQLite."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Inicializar banco de dados e criar tabelas."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Habilitar WAL para melhor concorrência
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        
        # Criar tabela principal
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                normalized_url TEXT NOT NULL UNIQUE,
                title TEXT,
                snippet TEXT,
                html_blob BLOB,
                headers_json TEXT,
                tls_json TEXT,
                heuristics_json TEXT,
                score REAL DEFAULT 0.0,
                discovered_at TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Criar índice para busca por URL
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pages_normalized_url 
            ON pages(normalized_url)
        """)
        
        # Criar índice para score
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pages_score 
            ON pages(score DESC)
        """)
        
        # Criar tabela FTS5 para busca de texto completo
        # SQLite FTS5 requer que a tabela virtual seja criada separadamente
        try:
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts 
                USING fts5(
                    title,
                    snippet,
                    html_content,
                    content='pages',
                    content_rowid='id'
                )
            """)
        except sqlite3.OperationalError as e:
            # Se já existir com estrutura diferente, recriar
            if "already exists" in str(e):
                logger.warning("Tabela FTS5 já existe, pulando criação")
            else:
                raise
        
        # Criar triggers para manter FTS sincronizado
        self._create_fts_triggers()
        
        self.conn.commit()
        logger.info(f"Banco de dados inicializado: {self.db_path}")
    
    def _create_fts_triggers(self):
        """Criar triggers para sincronizar FTS5."""
        # Trigger para INSERT
        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
                INSERT INTO pages_fts(rowid, title, snippet, html_content)
                VALUES (
                    new.id,
                    new.title,
                    new.snippet,
                    -- Extrair texto do HTML para indexação
                    COALESCE(
                        (SELECT text FROM (
                            -- Esta é uma simplificação; em produção, usaríamos um parser
                            SELECT substr(CAST(new.html_blob AS TEXT), 1, 10000) as text
                        )),
                        ''
                    )
                );
            END;
        """)
        
        # Trigger para UPDATE
        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pages BEGIN
                UPDATE pages_fts
                SET title = new.title,
                    snippet = new.snippet,
                    html_content = COALESCE(
                        (SELECT text FROM (
                            SELECT substr(CAST(new.html_blob AS TEXT), 1, 10000) as text
                        )),
                        ''
                    )
                WHERE rowid = old.id;
            END;
        """)
        
        # Trigger para DELETE
        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
                DELETE FROM pages_fts WHERE rowid = old.id;
            END;
        """)
    
    def upsert_page(self, record: Dict[str, Any]) -> int:
        """
        Inserir ou atualizar página.
        
        Args:
            record: Dicionário com dados da página
            
        Returns:
            ID da página inserida/atualizada
        """
        # Verificar se já existe pela URL normalizada
        normalized_url = record.get("normalized_url", "")
        existing_id = self.find_by_normalized_url(normalized_url)
        
        if existing_id:
            # Atualizar existente
            return self._update_page(existing_id, record)
        else:
            # Inserir novo
            return self._insert_page(record)
    
    def _insert_page(self, record: Dict[str, Any]) -> int:
        """Inserir nova página."""
        cursor = self.conn.execute("""
            INSERT INTO pages (
                url, normalized_url, title, snippet, html_blob,
                headers_json, tls_json, heuristics_json, score, discovered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get("url"),
            record.get("normalized_url"),
            record.get("title", "")[:500],
            record.get("snippet", "")[:500],
            record.get("html_blob", b""),
            record.get("headers_json", "{}"),
            record.get("tls_json", "{}"),
            record.get("heuristics_json", "{}"),
            record.get("score", 0.0),
            record.get("discovered_at", datetime.utcnow().isoformat())
        ))
        
        page_id = cursor.lastrowid
        self.conn.commit()
        return page_id
    
    def _update_page(self, page_id: int, record: Dict[str, Any]) -> int:
        """Atualizar página existente."""
        cursor = self.conn.execute("""
            UPDATE pages SET
                url = ?,
                title = ?,
                snippet = ?,
                html_blob = ?,
                headers_json = ?,
                tls_json = ?,
                heuristics_json = ?,
                score = ?,
                discovered_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            record.get("url"),
            record.get("title", "")[:500],
            record.get("snippet", "")[:500],
            record.get("html_blob", b""),
            record.get("headers_json", "{}"),
            record.get("tls_json", "{}"),
            record.get("heuristics_json", "{}"),
            record.get("score", 0.0),
            record.get("discovered_at", datetime.utcnow().isoformat()),
            page_id
        ))
        
        self.conn.commit()
        return page_id
    
    def get_page(self, page_id: int) -> Optional[Dict[str, Any]]:
        """Obter página pelo ID."""
        cursor = self.conn.execute("""
            SELECT * FROM pages WHERE id = ?
        """, (page_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return dict(row)
    
    def find_by_normalized_url(self, normalized_url: str) -> Optional[int]:
        """Encontrar ID pela URL normalizada."""
        cursor = self.conn.execute("""
            SELECT id FROM pages WHERE normalized_url = ?
        """, (normalized_url,))
        
        row = cursor.fetchone()
        return row[0] if row else None
    
    def search(
        self,
        query: str,
        filters: Dict[str, Any],
        limit: int = 10,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Buscar páginas.
        
        Args:
            query: Termos de busca
            filters: Filtros aplicados
            limit: Limite de resultados
            offset: Offset para paginação
            
        Returns:
            Tupla (resultados, total)
        """
        # Construir WHERE clause dinamicamente
        where_clauses = ["1=1"]
        params = []
        
        # Busca de texto completo
        if query and query.strip():
            where_clauses.append("pages_fts MATCH ?")
            params.append(f"{query}*")
        
        # Aplicar filtros
        if filters.get("tls") == "none":
            where_clauses.append("(tls_json = '{}' OR tls_json IS NULL OR tls_json = 'null')")
        
        if filters.get("has_forms") == "true":
            where_clauses.append("heuristics_json LIKE '%\"forms\"%'")
        
        if "min_score" in filters:
            where_clauses.append("score >= ?")
            params.append(float(filters["min_score"]))
        
        # Construir query SQL
        where_sql = " AND ".join(where_clauses)
        
        # Contar total (sem LIMIT/OFFSET)
        count_query = f"""
            SELECT COUNT(*) FROM pages
            WHERE id IN (
                SELECT rowid FROM pages_fts WHERE {where_sql}
            )
        """
        
        cursor = self.conn.execute(count_query, params)
        total = cursor.fetchone()[0]
        
        # Buscar resultados
        search_query = f"""
            SELECT pages.*, 
                   snippet(pages_fts, 0, '<b>', '</b>', '...', 64) as snippet_highlight
            FROM pages
            JOIN pages_fts ON pages.id = pages_fts.rowid
            WHERE {where_sql}
            ORDER BY pages.score DESC, pages.discovered_at DESC
            LIMIT ? OFFSET ?
        """
        
        params_extended = params + [limit, offset]
        cursor = self.conn.execute(search_query, params_extended)
        
        results = []
        for row in cursor:
            result = dict(row)
            
            # Parsear JSON fields
            for field in ["headers_json", "tls_json", "heuristics_json"]:
                if result.get(field):
                    try:
                        result[field] = json.loads(result[field])
                    except:
                        result[field] = {}
            
            results.append(result)
        
        return results, total
    
    def close(self):
        """Fechar conexão com banco de dados."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
''',

    "searcher.py": '''#!/usr/bin/env python3
"""
Módulo de busca com ranking e filtros.
"""
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from .engine import EngineContext

logger = logging.getLogger(__name__)

@dataclass
class SearchPage:
    """Página de resultados de busca."""
    results: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    total_pages: int
    
    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages
    
    @property
    def has_prev(self) -> bool:
        return self.page > 1

async def search(
    query: str,
    filters: Dict[str, Any],
    page: int = 1,
    page_size: int = 10,
    ctx: Optional[EngineContext] = None
) -> SearchPage:
    """
    Executar busca com ranking.
    
    Args:
        query: Termos de busca
        filters: Filtros aplicados
        page: Número da página
        page_size: Tamanho da página
        ctx: Contexto do engine
        
    Returns:
        SearchPage com resultados
    """
    # Validar entrada
    if page < 1:
        page = 1
    
    if page_size < 1 or page_size > 100:
        page_size = 10
    
    # Calcular offset
    offset = (page - 1) * page_size
    
    # Executar busca no storage
    results, total = ctx.storage.search(query, filters, page_size, offset)
    
    # Aplicar ranking adicional se necessário
    if query and query.strip():
        results = apply_ranking(results, query)
    
    # Calcular total de páginas
    total_pages = math.ceil(total / page_size) if total > 0 else 1
    
    return SearchPage(
        results=results,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )

def apply_ranking(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Aplicar ranking personalizado aos resultados.
    
    Args:
        results: Resultados da busca FTS
        query: Termos de busca
        
    Returns:
        Resultados reordenados
    """
    if not results:
        return results
    
    query_terms = query.lower().split()
    
    for result in results:
        # Calcular score baseado em múltiplos fatores
        fts_score = result.get("score", 0.0)
        heuristics_score = result.get("heuristics_json", {}).get("score", 0.0)
        
        # Score de recência (decai com o tempo)
        recency_score = calculate_recency_score(result.get("discovered_at"))
        
        # Score de relevância textual
        text_score = calculate_text_relevance(result, query_terms)
        
        # Combinar scores
        final_score = (
            0.4 * fts_score +
            0.3 * heuristics_score +
            0.2 * recency_score +
            0.1 * text_score
        )
        
        result["final_score"] = final_score
    
    # Ordenar por score final
    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    
    return results

def calculate_recency_score(discovered_at: str) -> float:
    """Calcular score baseado na recência."""
    try:
        if not discovered_at:
            return 0.5
        
        # Converter para datetime
        if discovered_at.endswith("Z"):
            discovered_at = discovered_at[:-1]
        
        dt = datetime.fromisoformat(discovered_at)
        now = datetime.utcnow()
        
        # Diferença em dias
        diff_days = (now - dt).total_seconds() / (24 * 3600)
        
        # Decaimento exponencial: 1.0 para agora, ~0.5 para 30 dias
        return max(0.1, math.exp(-diff_days / 30))
    except:
        return 0.5

def calculate_text_relevance(result: Dict[str, Any], query_terms: List[str]) -> float:
    """Calcular relevância textual simples."""
    score = 0.0
    
    # Verificar título
    title = result.get("title", "").lower()
    for term in query_terms:
        if term in title:
            score += 0.3
    
    # Verificar snippet
    snippet = result.get("snippet", "").lower()
    for term in query_terms:
        if term in snippet:
            score += 0.1
    
    # Normalizar para [0, 1]
    return min(1.0, score)
''',

    "heuristics.py": '''#!/usr/bin/env python3
"""
Heurísticas para detecção de problemas de segurança.
"""
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime

def analyze(parsed_page: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Analisar página e headers para detectar problemas.
    
    Args:
        parsed_page: Resultado do parser HTML
        headers: Headers HTTP
        
    Returns:
        Dicionário com tags e score
    """
    tags = []
    
    # Verificar TLS/SSL
    tls_issues = check_tls_issues(headers)
    tags.extend(tls_issues)
    
    # Verificar formulários
    form_issues = check_form_issues(parsed_page.get("forms", []))
    tags.extend(form_issues)
    
    # Verificar mensagens de erro
    error_issues = check_error_messages(parsed_page)
    tags.extend(error_issues)
    
    # Verificar paths administrativos
    admin_issues = check_admin_paths(parsed_page.get("links", []))
    tags.extend(admin_issues)
    
    # Verificar redirecionamentos abertos
    redirect_issues = check_open_redirects(parsed_page.get("links", []))
    tags.extend(redirect_issues)
    
    # Verificar listagem de diretórios
    dir_listing_issues = check_directory_listing(parsed_page)
    tags.extend(dir_listing_issues)
    
    # Verificar segredos em comentários
    secret_issues = check_secrets_in_comments(parsed_page.get("comments", []))
    tags.extend(secret_issues)
    
    # Calcular score baseado nas tags
    score = calculate_score(tags)
    
    return {
        "tags": list(set(tags)),  # Remover duplicatas
        "score": score
    }

def check_tls_issues(headers: Dict[str, str]) -> List[str]:
    """Verificar problemas relacionados a TLS/SSL."""
    issues = []
    
    # Verificar se é HTTP (não HTTPS)
    # Esta verificação é feita no nível do fetch_result
    # Aqui verificamos headers relacionados
    if headers.get("strict-transport-security"):
        issues.append("hsts_enabled")
    else:
        issues.append("no_hsts")
    
    return issues

def check_form_issues(forms: List[Dict[str, Any]]) -> List[str]:
    """Verificar problemas em formulários."""
    issues = []
    
    for form in forms:
        method = form.get("method", "GET").upper()
        has_csrf = form.get("has_csrf_token", False)
        
        if method in ["POST", "PUT", "DELETE"] and not has_csrf:
            issues.append("form_missing_csrf")
        
        # Verificar action vazia ou self-submit
        action = form.get("action", "").strip()
        if not action or action.startswith("#"):
            issues.append("form_self_submit")
    
    return issues

def check_error_messages(parsed_page: Dict[str, Any]) -> List[str]:
    """Detectar mensagens de erro expostas."""
    error_patterns = [
        r"error\s*:\s*\w+",
        r"exception",
        r"stack\s*trace",
        r"sql.*error",
        r"syntax\s*error",
        r"database\s*error",
        r"undefined",
        r"null\s*pointer",
        r"segmentation\s*fault",
        r"warning:",
        r"notice:",
        r"deprecated",
    ]
    
    issues = []
    text_to_check = ""
    
    # Combinar texto da página
    if parsed_page.get("first_text_blocks"):
        text_to_check += " ".join(parsed_page["first_text_blocks"]).lower()
    
    if parsed_page.get("title"):
        text_to_check += " " + parsed_page["title"].lower()
    
    # Buscar padrões
    for pattern in error_patterns:
        if re.search(pattern, text_to_check, re.IGNORECASE):
            issues.append("error_message_exposed")
            break
    
    return issues

def check_admin_paths(links: List[Dict[str, Any]]) -> List[str]:
    """Detectar paths administrativos."""
    admin_patterns = [
        r"/admin",
        r"/wp-admin",
        r"/administrator",
        r"/backend",
        r"/dashboard",
        r"/manager",
        r"/system",
        r"/cpanel",
        r"/phpmyadmin",
        r"/mysql",
        r"/pgadmin",
        r"/webmin",
    ]
    
    issues = []
    
    for link in links:
        href = link.get("href", "").lower()
        for pattern in admin_patterns:
            if re.search(pattern, href):
                issues.append("admin_path_exposed")
                break
    
    return list(set(issues))

def check_open_redirects(links: List[Dict[str, Any]]) -> List[str]:
    """Detectar possíveis redirecionamentos abertos."""
    redirect_params = ["url", "redirect", "next", "return", "r", "u"]
    issues = []
    
    for link in links:
        href = link.get("href", "")
        if "?" in href:
            # Verificar parâmetros de query
            query_part = href.split("?", 1)[1]
            params = query_part.split("&")
            
            for param in params:
                if "=" in param:
                    key, value = param.split("=", 1)
                    if key.lower() in redirect_params:
                        # Verificar se o valor é uma URL externa
                        if re.match(r"https?://", value, re.IGNORECASE):
                            issues.append("open_redirect_candidate")
                            break
    
    return issues

def check_directory_listing(parsed_page: Dict[str, Any]) -> List[str]:
    """Detectar listagem de diretórios."""
    indicators = [
        "index of /",
        "directory listing for",
        "parent directory",
        "[dir]",
        "[to parent directory]",
        "last modified",
        "size",
        "name",
        "description",
    ]
    
    issues = []
    text_to_check = ""
    
    # Combinar texto da página
    if parsed_page.get("title"):
        text_to_check += " " + parsed_page["title"].lower()
    
    if parsed_page.get("first_text_blocks"):
        text_to_check += " ".join(parsed_page["first_text_blocks"]).lower()
    
    # Buscar indicadores
    for indicator in indicators:
        if indicator in text_to_check:
            issues.append("directory_listing")
            break
    
    return issues

def check_secrets_in_comments(comments: List[str]) -> List[str]:
    """Detectar possíveis segredos em comentários HTML."""
    secret_patterns = [
        r"api[_-]?key\s*[:=]\s*[\"'][\w\-]{10,}[\"']",
        r"secret[_-]?key\s*[:=]\s*[\"'][\w\-]{10,}[\"']",
        r"password\s*[:=]\s*[\"'][\w\-]{6,}[\"']",
        r"token\s*[:=]\s*[\"'][\w\-]{10,}[\"']",
        r"aws[_-]?access[_-]?key",
        r"aws[_-]?secret[_-]?key",
        r"database[_-]?password",
        r"private[_-]?key",
        r"BEGIN (RSA|DSA|EC) PRIVATE KEY",
    ]
    
    issues = []
    
    for comment in comments:
        for pattern in secret_patterns:
            if re.search(pattern, comment, re.IGNORECASE):
                issues.append("possible_secret_in_comment")
                break
    
    return issues

def calculate_score(tags: List[str]) -> float:
    """Calcular score baseado nas tags encontradas."""
    # Pesos para diferentes tipos de issues
    weights = {
        "no_hsts": 0.2,
        "form_missing_csrf": 0.3,
        "error_message_exposed": 0.4,
        "admin_path_exposed": 0.3,
        "open_redirect_candidate": 0.2,
        "directory_listing": 0.3,
        "possible_secret_in_comment": 0.5,
    }
    
    # Score base é 1.0 (perfeito)
    score = 1.0
    
    # Subtrair baseado nas issues encontradas
    for tag in tags:
        if tag in weights:
            score -= weights[tag]
    
    # Garantir score entre 0.0 e 1.0
    return max(0.0, min(1.0, score))
''',

    "inspect.py": '''#!/usr/bin/env python3
"""
Renderizador seguro de HTML no terminal.
"""
import subprocess
import tempfile
import os
import re
import html
from typing import Optional
import logging
from .config import get_config

logger = logging.getLogger(__name__)

def sanitize_html(html_content: str, max_bytes: int = 1000000) -> str:
    """
    Sanitizar HTML removendo elementos perigosos.
    
    Args:
        html_content: HTML original
        max_bytes: Tamanho máximo do HTML sanitizado
        
    Returns:
        HTML sanitizado e seguro
    """
    # Truncar se muito grande
    if len(html_content) > max_bytes:
        html_content = html_content[:max_bytes]
        html_content += "\n<!-- [TRUNCATED] -->"
    
    # Remover scripts
    html_content = re.sub(r'<script\b[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remover iframes
    html_content = re.sub(r'<iframe\b[^>]*>.*?</iframe>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remover objetos
    html_content = re.sub(r'<object\b[^>]*>.*?</object>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remover applets
    html_content = re.sub(r'<applet\b[^>]*>.*?</applet>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remover event handlers (onclick, onload, etc.)
    event_handlers = [
        'onabort', 'onactivate', 'onafterprint', 'onafterupdate',
        'onbeforeactivate', 'onbeforecopy', 'onbeforecut', 'onbeforedeactivate',
        'onbeforeeditfocus', 'onbeforepaste', 'onbeforeprint', 'onbeforeunload',
        'onbeforeupdate', 'onblur', 'onbounce', 'oncellchange', 'onchange',
        'onclick', 'oncontextmenu', 'oncontrolselect', 'oncopy', 'oncut',
        'ondataavailable', 'ondatasetchanged', 'ondatasetcomplete',
        'ondblclick', 'ondeactivate', 'ondrag', 'ondragend', 'ondragenter',
        'ondragleave', 'ondragover', 'ondragstart', 'ondrop', 'onerror',
        'onerrorupdate', 'onfilterchange', 'onfinish', 'onfocus', 'onfocusin',
        'onfocusout', 'onhelp', 'onkeydown', 'onkeypress', 'onkeyup',
        'onlayoutcomplete', 'onload', 'onlosecapture', 'onmousedown',
        'onmouseenter', 'onmouseleave', 'onmousemove', 'onmouseout',
        'onmouseover', 'onmouseup', 'onmousewheel', 'onmove', 'onmoveend',
        'onmovestart', 'onpaste', 'onpropertychange', 'onreadystatechange',
        'onreset', 'onresize', 'onresizeend', 'onresizestart', 'onrowenter',
        'onrowexit', 'onrowsdelete', 'onrowsinserted', 'onscroll',
        'onselect', 'onselectionchange', 'onselectstart', 'onstart',
        'onstop', 'onsubmit', 'onunload'
    ]
    
    for handler in event_handlers:
        html_content = re.sub(
            rf'{handler}\s*=\s*["\'][^"\']*["\']',
            '',
            html_content,
            flags=re.IGNORECASE
        )
        html_content = re.sub(
            rf'{handler}\s*=\s*[^ >]+',
            '',
            html_content,
            flags=re.IGNORECASE
        )
    
    # Remover data: URIs muito grandes
    html_content = re.sub(
        r'src\s*=\s*["\']data:[^"\']{1000,}["\']',
        'src=""',
        html_content,
        flags=re.IGNORECASE
    )
    
    # Remover javascript: URIs
    html_content = re.sub(
        r'href\s*=\s*["\']javascript:[^"\']*["\']',
        'href="#"',
        html_content,
        flags=re.IGNORECASE
    )
    
    # Remover meta refresh
    html_content = re.sub(
        r'<meta[^>]*http-equiv\s*=\s*["\']refresh["\'][^>]*>',
        '',
        html_content,
        flags=re.IGNORECASE
    )
    
    # Adicionar aviso de segurança no topo
    warning = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>BareNet Safe Render</title>
    <style>
        body { font-family: monospace; margin: 20px; }
        .warning { 
            background-color: #fff3cd; 
            border: 1px solid #ffeaa7;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 5px;
            color: #856404;
        }
        pre { white-space: pre-wrap; word-wrap: break-word; }
    </style>
</head>
<body>
    <div class="warning">
        <strong>⚠️  RENDERIZAÇÃO SEGURA Barenet</strong><br>
        JavaScript, iframes e conteúdo ativo foram removidos.<br>
        Esta é uma visualização estática e segura.
    </div>
"""
    
    # Garantir que temos tags HTML básicas
    if not re.search(r'<html[^>]*>', html_content, re.IGNORECASE):
        html_content = warning + html_content + "</body></html>"
    else:
        # Inserir aviso após <body>
        html_content = re.sub(
            r'(<body[^>]*>)',
            r'\\1' + warning,
            html_content,
            flags=re.IGNORECASE
        )
    
    return html_content

def render_in_terminal(html_content: str, renderer: str = "w3m", url: str = "") -> None:
    """
    Renderizar HTML no terminal usando renderizador externo.
    
    Args:
        html_content: HTML a ser renderizado
        renderer: Renderizador a usar (w3m, links, lynx)
        url: URL original para exibir no cabeçalho
    """
    config = get_config()
    
    # Sanitizar HTML
    safe_html = sanitize_html(html_content, config.MAX_RENDER_BYTES)
    
    # Verificar se renderizador está disponível
    renderer_cmd = find_renderer(renderer)
    if not renderer_cmd:
        logger.error(f"Renderizador '{renderer}' não encontrado.")
        print("Renderizadores disponíveis: w3m, links, lynx")
        print("Instale um deles ou use --renderer diferente.")
        return
    
    # Criar arquivo temporário com HTML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(safe_html)
        temp_file = f.name
    
    try:
        # Comando para renderizar
        cmd = [renderer_cmd]
        
        # Opções específicas por renderizador
        if renderer == "w3m":
            cmd.extend(["-T", "text/html", "-dump", temp_file])
        elif renderer == "links":
            cmd.extend(["-dump", temp_file])
        elif renderer == "lynx":
            cmd.extend(["-dump", "-nolist", temp_file])
        
        # Executar renderizador
        print(f"\\n🔍  Inspecionando: {url}\\n" + "="*80)
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Erro ao renderizar: {result.stderr}")
            # Fallback: mostrar HTML sanitizado como texto
            print("\\n=== FALLBACK: HTML SANITIZADO (texto simples) ===\\n")
            print(re.sub(r'<[^>]+>', '', safe_html[:2000]))
    
    except Exception as e:
        logger.error(f"Erro durante renderização: {e}")
        print(f"Erro: {e}")
    
    finally:
        # Limpar arquivo temporário
        try:
            os.unlink(temp_file)
        except:
            pass
        
        print("\\n" + "="*80)
        print("Pressione Enter para continuar...", end="")
        input()

def find_renderer(preferred: str) -> Optional[str]:
    """
    Encontrar renderizador disponível no sistema.
    
    Args:
        preferred: Renderizador preferido
        
    Returns:
        Caminho para o executável ou None
    """
    # Lista de renderizadores em ordem de preferência
    renderers = [preferred, "w3m", "links", "lynx", "elinks"]
    
    for renderer in renderers:
        try:
            result = subprocess.run(
                ["which", renderer],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return renderer
        except:
            continue
    
    return None
''',

    "cache.py": '''#!/usr/bin/env python3
"""
Sistema de cache local para resultados HTTP.
"""
import sqlite3
import json
import gzip
import hashlib
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, NamedTuple
from pathlib import Path
import logging
from .config import Config

logger = logging.getLogger(__name__)

class CachedRecord(NamedTuple):
    """Registro em cache."""
    url: str
    headers: Dict[str, str]
    body: bytes
    tls_info: Optional[Dict[str, Any]]
    cached_at: datetime
    expires_at: datetime

class Cache:
    """Gerenciador de cache local."""
    
    def __init__(self, cache_dir: str, ttl_days: int = 30):
        self.cache_dir = Path(cache_dir)
        self.ttl_days = ttl_days
        
        # Garantir que diretório existe
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Caminho para banco de dados de metadados
        self.db_path = self.cache_dir / "cache_index.db"
        self._init_db()
    
    def _init_db(self):
        """Inicializar banco de dados de cache."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                url_hash TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                headers_path TEXT NOT NULL,
                body_path TEXT NOT NULL,
                tls_info_path TEXT,
                cached_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                last_accessed TIMESTAMP NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        # Índices para busca eficiente
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_expires 
            ON cache_entries(expires_at)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_url 
            ON cache_entries(url)
        """)
        
        self.conn.commit()
    
    def _get_url_hash(self, url: str) -> str:
        """Calcular hash SHA256 da URL."""
        return hashlib.sha256(url.encode("utf-8")).hexdigest()
    
    def get(self, url: str) -> Optional[CachedRecord]:
        """
        Obter registro do cache.
        
        Args:
            url: URL para buscar
            
        Returns:
            CachedRecord se encontrado e não expirado, None caso contrário
        """
        url_hash = self._get_url_hash(url)
        
        cursor = self.conn.execute("""
            SELECT * FROM cache_entries 
            WHERE url_hash = ? AND expires_at > ?
        """, (url_hash, datetime.utcnow().isoformat()))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        try:
            # Atualizar estatísticas de acesso
            self.conn.execute("""
                UPDATE cache_entries 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE url_hash = ?
            """, (datetime.utcnow().isoformat(), url_hash))
            self.conn.commit()
            
            # Carregar dados dos arquivos
            headers = self._load_json_file(row["headers_path"])
            body = self._load_gzipped_file(row["body_path"])
            tls_info = None
            if row["tls_info_path"]:
                tls_info = self._load_json_file(row["tls_info_path"])
            
            return CachedRecord(
                url=row["url"],
                headers=headers,
                body=body,
                tls_info=tls_info,
                cached_at=datetime.fromisoformat(row["cached_at"]),
                expires_at=datetime.fromisoformat(row["expires_at"])
            )
        except Exception as e:
            logger.error(f"Erro ao carregar cache para {url}: {e}")
            # Se houver erro ao carregar, remover entrada corrompida
            self.invalidate(url)
            return None
    
    def put(self, url: str, headers: Dict[str, str], body: bytes, tls_info: Optional[Dict[str, Any]] = None):
        """
        Armazenar registro no cache.
        
        Args:
            url: URL
            headers: Headers HTTP
            body: Corpo da resposta
            tls_info: Informações TLS (opcional)
        """
        try:
            url_hash = self._get_url_hash(url)
            
            # Gerar caminhos de arquivo
            base_name = url_hash
            headers_path = self.cache_dir / f"{base_name}_headers.json.gz"
            body_path = self.cache_dir / f"{base_name}_body.html.gz"
            tls_info_path = None
            if tls_info:
                tls_info_path = self.cache_dir / f"{base_name}_tls.json.gz"
            
            # Salvar arquivos
            self._save_json_gzipped(headers_path, headers)
            self._save_gzipped(body_path, body)
            if tls_info and tls_info_path:
                self._save_json_gzipped(tls_info_path, tls_info)
            
            # Calcular datas
            now = datetime.utcnow()
            expires_at = now + timedelta(days=self.ttl_days)
            
            # Inserir/atualizar no banco
            self.conn.execute("""
                INSERT OR REPLACE INTO cache_entries 
                (url_hash, url, headers_path, body_path, tls_info_path, 
                 cached_at, expires_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                url_hash,
                url,
                str(headers_path),
                str(body_path),
                str(tls_info_path) if tls_info_path else None,
                now.isoformat(),
                expires_at.isoformat(),
                now.isoformat(),
                1
            ))
            
            self.conn.commit()
            logger.debug(f"Cache armazenado para {url}")
            
        except Exception as e:
            logger.error(f"Erro ao armazenar cache para {url}: {e}")
            # Tentar limpar arquivos criados em caso de erro
            self._cleanup_files(url_hash)
    
    def invalidate(self, url: str):
        """Invalidar entrada de cache."""
        url_hash = self._get_url_hash(url)
        
        # Buscar caminhos de arquivo
        cursor = self.conn.execute(
            "SELECT headers_path, body_path, tls_info_path FROM cache_entries WHERE url_hash = ?",
            (url_hash,)
        )
        row = cursor.fetchone()
        
        if row:
            # Remover arquivos
            paths = [row["headers_path"], row["body_path"]]
            if row["tls_info_path"]:
                paths.append(row["tls_info_path"])
            
            for path_str in paths:
                try:
                    if path_str:
                        Path(path_str).unlink(missing_ok=True)
                except:
                    pass
        
        # Remover do banco
        self.conn.execute("DELETE FROM cache_entries WHERE url_hash = ?", (url_hash,))
        self.conn.commit()
    
    def prune_expired(self) -> int:
        """Remover entradas expiradas do cache."""
        cursor = self.conn.execute(
            "SELECT url_hash, headers_path, body_path, tls_info_path FROM cache_entries WHERE expires_at <= ?",
            (datetime.utcnow().isoformat(),)
        )
        
        expired_rows = cursor.fetchall()
        count = len(expired_rows)
        
        for row in expired_rows:
            # Remover arquivos
            paths = [row["headers_path"], row["body_path"]]
            if row["tls_info_path"]:
                paths.append(row["tls_info_path"])
            
            for path_str in paths:
                try:
                    if path_str:
                        Path(path_str).unlink(missing_ok=True)
                except:
                    pass
        
        # Remover do banco
        self.conn.execute("DELETE FROM cache_entries WHERE expires_at <= ?", (datetime.utcnow().isoformat(),))
        self.conn.commit()
        
        logger.info(f"Cache limpo: {count} entradas expiradas removidas")
        return count
    
    def _save_json_gzipped(self, path: Path, data: Dict[str, Any]):
        """Salvar dados JSON em arquivo gzipped."""
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=None)
    
    def _save_gzipped(self, path: Path, data: bytes):
        """Salvar dados binários em arquivo gzipped."""
        with gzip.open(path, "wb") as f:
            f.write(data)
    
    def _load_json_file(self, path_str: str) -> Dict[str, Any]:
        """Carregar dados JSON de arquivo gzipped."""
        with gzip.open(path_str, "rt", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_gzipped_file(self, path_str: str) -> bytes:
        """Carregar dados binários de arquivo gzipped."""
        with gzip.open(path_str, "rb") as f:
            return f.read()
    
    def _cleanup_files(self, url_hash: str):
        """Limpar arquivos de cache em caso de erro."""
        base_name = url_hash
        patterns = [
            f"{base_name}_headers.json.gz",
            f"{base_name}_body.html.gz",
            f"{base_name}_tls.json.gz",
        ]
        
        for pattern in patterns:
            try:
                (self.cache_dir / pattern).unlink(missing_ok=True)
            except:
                pass
    
    def close(self):
        """Fechar conexão com banco."""
        if hasattr(self, "conn"):
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
''',

    "exporters.py": '''#!/usr/bin/env python3
"""
Exportadores para diferentes formatos.
"""
import json
import csv
from typing import List, Dict, Any, Iterator, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def export_json(path: str, records: List[Dict[str, Any]], indent: int = 2) -> None:
    """
    Exportar registros para JSON.
    
    Args:
        path: Caminho do arquivo de saída
        records: Lista de registros
        indent: Indentação do JSON
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=indent, ensure_ascii=False, default=str)
        logger.info(f"Exportado {len(records)} registros para {path}")
    except Exception as e:
        logger.error(f"Erro ao exportar JSON: {e}")
        raise

def export_csv(path: str, records: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    """
    Exportar registros para CSV.
    
    Args:
        path: Caminho do arquivo de saída
        records: Lista de registros
        fields: Campos a exportar (todos se None)
    """
    if not records:
        logger.warning("Nenhum registro para exportar")
        return
    
    # Determinar campos se não especificados
    if not fields:
        fields = list(records[0].keys())
    
    try:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            
            for record in records:
                # Preparar linha (converter valores complexos para string)
                row = {}
                for field in fields:
                    value = record.get(field)
                    if isinstance(value, (dict, list)):
                        row[field] = json.dumps(value, ensure_ascii=False)
                    elif value is None:
                        row[field] = ""
                    else:
                        row[field] = str(value)
                writer.writerow(row)
        
        logger.info(f"Exportado {len(records)} registros para {path}")
    except Exception as e:
        logger.error(f"Erro ao exportar CSV: {e}")
        raise

def export_ndjson(path: str, records: List[Dict[str, Any]]) -> None:
    """
    Exportar registros para NDJSON (Newline Delimited JSON).
    
    Args:
        path: Caminho do arquivo de saída
        records: Lista de registros
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False, default=str)
                f.write("\\n")
        
        logger.info(f"Exportado {len(records)} registros para {path}")
    except Exception as e:
        logger.error(f"Erro ao exportar NDJSON: {e}")
        raise

def export_html_blob_to_file(record: Dict[str, Any], output_dir: str) -> str:
    """
    Exportar HTML blob para arquivo separado.
    
    Args:
        record: Registro com html_blob
        output_dir: Diretório de saída
        
    Returns:
        Caminho para o arquivo criado
    """
    html_blob = record.get("html_blob")
    if not html_blob:
        raise ValueError("Registro não contém html_blob")
    
    # Criar diretório se não existir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Gerar nome de arquivo seguro
    page_id = record.get("id", "unknown")
    url_hash = hashlib.md5(record.get("url", "").encode()).hexdigest()[:8]
    filename = f"page_{page_id}_{url_hash}.html"
    filepath = Path(output_dir) / filename
    
    # Salvar HTML
    with open(filepath, "wb") as f:
        f.write(html_blob)
    
    return str(filepath)
''',

    "plugins/localfile.py": '''#!/usr/bin/env python3
"""
Plugin para carregar URLs de arquivo de texto.
"""
import re
from typing import Iterator, List
from urllib.parse import urlparse
from ..utils import normalize_url

def iter_urls_from_file(path: str) -> Iterator[str]:
    """
    Gerar URLs de um arquivo de texto.
    
    Args:
        path: Caminho para o arquivo
        
    Yields:
        URLs válidas
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Pular linhas vazias e comentários
                if not line or line.startswith("#"):
                    continue
                
                # Validar URL básica
                if is_valid_url(line):
                    yield line
                else:
                    print(f"Linha {line_num}: URL inválida ignorada: {line[:50]}")
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {path}")
        return []
    except Exception as e:
        print(f"Erro ao ler arquivo {path}: {e}")
        return []

def is_valid_url(url: str) -> bool:
    """Verificar se string parece uma URL válida."""
    # Padrões simples de URL
    url_patterns = [
        r'^https?://',  # http:// ou https://
        r'^ftp://',     # ftp://
        r'^file://',    # file://
    ]
    
    for pattern in url_patterns:
        if re.match(pattern, url, re.IGNORECASE):
            try:
                parsed = urlparse(url)
                if parsed.scheme and parsed.netloc:
                    return True
            except:
                pass
    
    return False

def dedupe_urls(urls: List[str]) -> List[str]:
    """
    Remover URLs duplicadas (considerando normalização).
    
    Args:
        urls: Lista de URLs
        
    Returns:
        Lista única de URLs
    """
    seen = set()
    unique = []
    
    for url in urls:
        normalized = normalize_url(url)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(url)
    
    return unique
''',

    "ui/list_view.py": '''#!/usr/bin/env python3
"""
Interface TUI para navegação de resultados.
"""
from prompt_toolkit import PromptSession
from prompt_toolkit.layout import Layout
from prompt_toolkit.widgets import Box, Frame, TextArea, Label
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import Application
from prompt_toolkit.layout.containers import VSplit, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from typing import Optional, List, Dict, Any
import asyncio
from ..engine import Engine
from ..searcher import SearchPage
import logging

logger = logging.getLogger(__name__)

class ListViewUI:
    """Interface TUI para navegação de resultados."""
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.current_page: Optional[SearchPage] = None
        self.selected_index = 0
        self.query = ""
        self.filters = {}
        
        # Widgets
        self.results_list = TextArea(
            text="",
            multiline=True,
            scrollbar=True,
            focusable=True,
            read_only=True
        )
        
        self.detail_view = TextArea(
            text="",
            multiline=True,
            scrollbar=True,
            focusable=False,
            read_only=True
        )
        
        self.status_bar = Label("")
        
        # Bindings
        self.bindings = KeyBindings()
        self._setup_bindings()
        
        # Layout
        self.layout = self._create_layout()
        
        # Aplicação
        self.app = Application(
            layout=self.layout,
            key_bindings=self.bindings,
            mouse_support=True,
            full_screen=True,
            style=Style.from_dict({
                "status": "reverse",
                "selected": "bg:#008800 #ffffff",
            })
        )
    
    def _create_layout(self):
        """Criar layout da interface."""
        return Layout(
            HSplit([
                # Barra de título
                Window(
                    height=1,
                    content=FormattedTextControl(
                        lambda: self._get_title_text()
                    ),
                    style="class:title",
                ),
                
                # Área principal
                VSplit([
                    # Lista de resultados
                    Frame(
                        title="Resultados",
                        body=self.results_list,
                        width=0.5,
                    ),
                    
                    # Detalhes
                    Frame(
                        title="Detalhes",
                        body=self.detail_view,
                    ),
                ]),
                
                # Barra de status
                Window(
                    height=1,
                    content=FormattedTextControl(
                        lambda: self._get_status_text()
                    ),
                    style="class:status",
                ),
            ])
        )
    
    def _get_title_text(self):
        """Obter texto da barra de título."""
        if self.current_page:
            return f" BareNet | Query: {self.query} | Página {self.current_page.page}/{self.current_page.total_pages} ({self.current_page.total} resultados) "
        return " BareNet | Digite uma query "
    
    def _get_status_text(self):
        """Obter texto da barra de status."""
        help_text = "↑↓: Navegar | Enter: Inspecionar | →←: Páginas | /: Buscar | e: Exportar | q: Sair"
        if self.current_page and self.current_page.results:
            selected = self.current_page.results[self.selected_index]
            url = selected.get("url", "")[:80]
            return f" {url} | {help_text}"
        return f" {help_text}"
    
    def _setup_bindings(self):
        """Configurar atalhos de teclado."""
        kb = self.bindings
        
        @kb.add("up")
        def _(event):
            if self.current_page and self.current_page.results:
                self.selected_index = max(0, self.selected_index - 1)
                self._update_display()
        
        @kb.add("down")
        def _(event):
            if self.current_page and self.current_page.results:
                self.selected_index = min(
                    len(self.current_page.results) - 1,
                    self.selected_index + 1
                )
                self._update_display()
        
        @kb.add("enter")
        def _(event):
            if self.current_page and self.current_page.results:
                self._inspect_selected()
        
        @kb.add("right")
        def _(event):
            if self.current_page and self.current_page.has_next:
                asyncio.create_task(self._load_page(self.current_page.page + 1))
        
        @kb.add("left")
        def _(event):
            if self.current_page and self.current_page.has_prev:
                asyncio.create_task(self._load_page(self.current_page.page - 1))
        
        @kb.add("/")
        def _(event):
            asyncio.create_task(self._prompt_search())
        
        @kb.add("e")
        def _(event):
            if self.current_page and self.current_page.results:
                asyncio.create_task(self._export_selected())
        
        @kb.add("q")
        def _(event):
            event.app.exit()
        
        @kb.add("c-c")
        def _(event):
            event.app.exit()
    
    def _update_display(self):
        """Atualizar exibição."""
        if not self.current_page or not self.current_page.results:
            self.results_list.text = "Nenhum resultado"
            self.detail_view.text = ""
            return
        
        # Atualizar lista de resultados
        lines = []
        for i, result in enumerate(self.current_page.results):
            prefix = "▶ " if i == self.selected_index else "  "
            title = result.get("title", "Sem título")[:50]
            score = result.get("score", 0)
            lines.append(f"{prefix}[{score:.2f}] {title}")
        
        self.results_list.text = "\\n".join(lines)
        
        # Atualizar detalhes
        selected = self.current_page.results[self.selected_index]
        details = self._format_details(selected)
        self.detail_view.text = details
    
    def _format_details(self, result: Dict[str, Any]) -> str:
        """Formatar detalhes do resultado."""
        lines = []
        
        # URL
        lines.append(f"URL: {result.get('url')}")
        lines.append("")
        
        # Título e snippet
        lines.append(f"Título: {result.get('title', 'N/A')}")
        if result.get("snippet"):
            lines.append(f"Snippet: {result.get('snippet')[:200]}")
        lines.append("")
        
        # Metadata
        lines.append(f"Score: {result.get('score', 0):.2f}")
        lines.append(f"Descoberto em: {result.get('discovered_at', 'N/A')}")
        lines.append("")
        
        # Heurísticas
        heuristics = result.get("heuristics_json", {})
        if isinstance(heuristics, dict):
            tags = heuristics.get("tags", [])
            if tags:
                lines.append(f"Tags: {', '.join(tags)}")
        
        # Headers (primeiros)
        headers = result.get("headers_json", {})
        if isinstance(headers, dict):
            lines.append("Headers:")
            for key in list(headers.keys())[:3]:
                lines.append(f"  {key}: {headers[key][:50]}")
        
        return "\\n".join(lines)
    
    async def _load_page(self, page: int):
        """Carregar página de resultados."""
        try:
            self.current_page = await self.engine.search(
                self.query,
                self.filters,
                page,
                10
            )
            self.selected_index = 0
            self._update_display()
        except Exception as e:
            logger.error(f"Erro ao carregar página: {e}")
    
    async def _prompt_search(self):
        """Solicitar nova busca ao usuário."""
        from prompt_toolkit import PromptSession
        
        session = PromptSession()
        new_query = await session.prompt_async(
            "Buscar: ",
            default=self.query
        )
        
        if new_query is not None and new_query != self.query:
            self.query = new_query
            await self._load_page(1)
    
    def _inspect_selected(self):
        """Inspecionar item selecionado."""
        if not self.current_page or not self.current_page.results:
            return
        
        selected = self.current_page.results[self.selected_index]
        page_id = selected.get("id")
        
        if page_id:
            # Suspender TUI temporariamente
            self.app.suspend()
            try:
                self.engine.inspect(str(page_id))
            finally:
                # Retomar TUI
                self.app.resume()
    
    async def _export_selected(self):
        """Exportar item selecionado."""
        if not self.current_page or not self.current_page.results:
            return
        
        selected = self.current_page.results[self.selected_index]
        page_id = selected.get("id")
        
        if page_id:
            # TODO: Implementar diálogo de exportação
            print(f"Exportar ID {page_id}")
    
    async def run(self, initial_query: str = ""):
        """Executar interface."""
        self.query = initial_query
        if self.query:
            await self._load_page(1)
        
        await self.app.run_async()

def run_list_ui(engine: Engine, initial_query: str = ""):
    """
    Executar interface TUI.
    
    Args:
        engine: Instância do Engine
        initial_query: Query inicial (opcional)
    """
    ui = ListViewUI(engine)
    asyncio.run(ui.run(initial_query))
''',

    "utils.py": '''#!/usr/bin/env python3
"""
Funções utilitárias gerais.
"""
import hashlib
import re
from urllib.parse import urlparse, urlunparse, urljoin, quote
from typing import Optional, List
from datetime import datetime
import unicodedata
import ipaddress

def normalize_url(url: str) -> str:
    """
    Normalizar URL para comparação e armazenamento.
    
    Args:
        url: URL a ser normalizada
        
    Returns:
        URL normalizada
    """
    try:
        parsed = urlparse(url)
        
        # Converter hostname para minúsculas
        netloc = parsed.netloc.lower()
        
        # Remover porta padrão
        if parsed.scheme == "http" and netloc.endswith(":80"):
            netloc = netloc[:-3]
        elif parsed.scheme == "https" and netloc.endswith(":443"):
            netloc = netloc[:-4]
        
        # Remover fragmento (#)
        # Remover query string (?)
        # Manter path como está (case-sensitive para alguns servidores)
        
        normalized = urlunparse((
            parsed.scheme,
            netloc,
            parsed.path,
            "",  # params
            "",  # query - removemos query string para normalização
            ""   # fragment
        ))
        
        return normalized.rstrip("/")
    except:
        # Em caso de erro, retornar original
        return url

def sha256_hash(text: str) -> str:
    """Calcular hash SHA256 de uma string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def iso_now() -> str:
    """Obter timestamp atual em formato ISO8601."""
    return datetime.utcnow().isoformat() + "Z"

def parse_iso(iso_string: str) -> datetime:
    """Parsear string ISO8601 para datetime."""
    # Remover 'Z' se presente
    if iso_string.endswith("Z"):
        iso_string = iso_string[:-1]
    
    # Adicionar microsegundos se não existirem
    if "." not in iso_string:
        iso_string += ".000000"
    
    return datetime.fromisoformat(iso_string)

def truncate_html(html: str, max_bytes: int) -> str:
    """
    Truncar HTML preservando tags abertas.
    
    Args:
        html: HTML a truncar
        max_bytes: Número máximo de bytes
        
    Returns:
        HTML truncado
    """
    if len(html) <= max_bytes:
        return html
    
    # Encontrar ponto de corte seguro (fim de tag)
    cut_point = max_bytes
    while cut_point > 0 and html[cut_point] != ">":
        cut_point -= 1
    
    if cut_point == 0:
        # Não encontrou '>', cortar no limite
        cut_point = max_bytes
    
    truncated = html[:cut_point + 1]
    
    # Fechar tags abertas
    open_tags = []
    tag_pattern = re.compile(r'<([a-zA-Z][a-zA-Z0-9]*)(?:\s[^>]*)?>')
    close_pattern = re.compile(r'</([a-zA-Z][a-zA-Z0-9]*)>')
    
    for match in tag_pattern.finditer(truncated):
        tag = match.group(1).lower()
        if tag not in ["br", "hr", "img", "input", "meta", "link"]:
            open_tags.append(tag)
    
    for match in close_pattern.finditer(truncated):
        tag = match.group(1).lower()
        if tag in open_tags:
            open_tags.remove(tag)
    
    # Adicionar tags de fechamento na ordem inversa
    for tag in reversed(open_tags):
        truncated += f"</{tag}>"
    
    truncated += "<!-- [TRUNCATED] -->"
    return truncated

def is_local_host(hostname: str) -> bool:
    """
    Verificar se hostname é local.
    
    Args:
        hostname: Hostname a verificar
        
    Returns:
        True se for host local
    """
    if not hostname:
        return False
    
    # Localhost
    if hostname.lower() in ["localhost", "localhost.localdomain"]:
        return True
    
    # Endereços de loopback
    if hostname.startswith("127.") or hostname == "::1":
        return True
    
    # IPs privados
    try:
        ip = ipaddress.ip_address(hostname)
        return ip.is_private
    except:
        pass
    
    # Domínios locais
    local_domains = [".local", ".localhost", ".internal", ".home", ".lan"]
    for domain in local_domains:
        if hostname.lower().endswith(domain):
            return True
    
    return False

def extract_domain(url: str) -> Optional[str]:
    """Extrair domínio de uma URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return None

def safe_filename(text: str, max_length: int = 100) -> str:
    """Criar nome de arquivo seguro a partir de texto."""
    # Remover caracteres não seguros
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r'\\s+', "_", text)
    text = text.strip("_.")
    
    # Truncar se muito longo
    if len(text) > max_length:
        text = text[:max_length]
    
    return text or "unnamed"
''',

    "requirements.txt": '''httpx[http2]>=0.24.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
prompt_toolkit>=3.0.0
aiofiles>=23.0.0
python-dotenv>=1.0.0
click>=8.1.0
''',

    "README.md": '''# BareNet

Scanner passivo de segurança web que indexa, analisa e expõe informações estruturais de aplicações web.

## Visão Geral

BareNet é uma ferramenta para segurança ofensiva e defensiva que permite:
- Indexar páginas web de forma passiva (sem interagir com formulários ou executar JavaScript)
- Analisar heurísticas de segurança (CSRF tokens faltando, mensagens de erro expostas, etc.)
- Buscar rapidamente no conteúdo indexado
- Inspecionar páginas de forma segura no terminal
- Exportar resultados para análise posterior

## Instalação

```bash
# Clonar repositório
git clone <repo-url>
cd barenet

# Instalar dependências
pip install -r requirements.txt

# Instalar renderizador (opcional, para funcionalidade inspect)
sudo apt-get install w3m  # ou links, lynx
