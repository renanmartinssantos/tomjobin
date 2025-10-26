import io
import zipfile
import csv
import json
import re
from typing import Dict, List, Tuple, Any, Optional
import unicodedata
import networkx as nx
import pandas as pd
from jobspy import scrape_jobs
from collections import Counter
import uuid
from pathlib import Path
import os
import math

STOPWORDS = {
    "a","acerca","adeus","agora","ainda","alem","algmas","algo","algumas","alguns","ali","além","ambas","ambos","ano","anos","antes","ao","aonde","aos","apenas","apoio","apontar","apos","após","aquela","aquelas","aquele","aqueles","aqui","aquilo","as","assim","através","atrás","até","aí","baixo","bastante","bem","boa","boas","bom","bons","breve","cada","caminho","catorze","cedo","cento","certamente","certeza","cima","cinco","coisa","com","como","comprido","conhecido","conselho","contra","contudo","corrente","cuja","cujas","cujo","cujos","custa","cá","da","daquela","daquelas","daquele","daqueles","dar","das","de","debaixo","dela","delas","dele","deles","demais","dentro","depois","desde","desligado","dessa","dessas","desse","desses","desta","destas","deste","destes","deve","devem","deverá","dez","dezanove","dezasseis","dezassete","dezoito","dia","diante","direita","dispoe","dispoem","diversa","diversas","diversos","diz","dizem","dizer","do","dois","dos","doze","duas","durante","dá","dão","dúvida","e","ela","elas","ele","eles","em","embora","enquanto","entao","entre","então","era","eram","essa","essas","esse","esses","esta","estado","estamos","estar","estará","estas","estava","estavam","este","esteja","estejam","estejamos","estes","esteve","estive","estivemos","estiver","estivera","estiveram","estiverem","estivermos","estivesse","estivessem","estiveste","estivestes","estivéramos","estivéssemos","estou","está","estás","estávamos","estão","eu","exemplo","falta","fará","favor","faz","fazeis","fazem","fazemos","fazer","fazes","fazia","faço","fez","fim","final","foi","fomos","for","fora","foram","forem","forma","formos","fosse","fossem","foste","fostes","fui","fôramos","fôssemos","geral","grande","grandes","grupo","ha","haja","hajam","hajamos","havemos","havia","hei","hoje","hora","horas","houve","houvemos","houver","houvera","houveram","houverei","houverem","houveremos","houveria","houveriam","houvermos","houverá","houverão","houveríamos","houvesse","houvessem","houvéramos","houvéssemos","há","hão","iniciar","inicio","ir","irá","isso","ista","iste","isto","já","lado","lhe","lhes","ligado","local","logo","longe","lugar","lá","maior","maioria","maiorias","mais","mal","mas","me","mediante","meio","menor","menos","meses","mesma","mesmas","mesmo","mesmos","meu","meus","mil","minha","minhas","momento","muito","muitos","máximo","mês","na","nada","nao","naquela","naquelas","naquele","naqueles","nas","nem","nenhuma","nessa","nessas","nesse","nesses","nesta","nestas","neste","nestes","no","noite","nome","nos","nossa","nossas","nosso","nossos","nova","novas","nove","novo","novos","num","numa","numas","nunca","nuns","não","nível","nós","número","o","obra","obrigada","obrigado","oitava","oitavo","oito","onde","ontem","onze","os","ou","outra","outras","outro","outros","para","parece","parte","partir","paucas","pegar","pela","pelas","pelo","pelos","perante","perto","pessoas","pode","podem","poder","poderá","podia","pois","ponto","pontos","por","porque","porquê","portanto","posição","possivelmente","posso","possível","pouca","pouco","poucos","povo","primeira","primeiras","primeiro","primeiros","promeiro","propios","proprio","própria","próprias","próprio","próprios","próxima","próximas","próximo","próximos","puderam","pôde","põe","põem","quais","qual","qualquer","quando","quanto","quarta","quarto","quatro","que","quem","quer","quereis","querem","queremas","queres","quero","questão","quieto","quinta","quinto","quinze","quáis","quê","relação","sabe","sabem","saber","se","segunda","segundo","sei","seis","seja","sejam","sejamos","sem","sempre","sendo","ser","serei","seremos","seria","seriam","será","serão","seríamos","sete","seu","seus","sexta","sexto","sim","sistema","sob","sobre","sois","somente","somos","sou","sua","suas","são","sétima","sétimo","só","tal","talvez","tambem","também","tanta","tantas","tanto","tarde","te","tem","temos","tempo","tendes","tenha","tenham","tenhamos","tenho","tens","tentar","tentaram","tente","tentei","ter","terceira","terceiro","terei","teremos","teria","teriam","terá","terão","teríamos","teu","teus","teve","tinha","tinham","tipo","tive","tivemos","tiver","tivera","tiveram","tiverem","tivermos","tivesse","tivessem","tiveste","tivestes","tivéramos","tivéssemos","toda","todas","todo","todos","trabalhar","trabalho","treze","três","tu","tua","tuas","tudo","tão","tém","têm","tínhamos","um","uma","umas","uns","usa","usar","vai","vais","valor","veja","vem","vens","ver","verdade","verdadeiro","vez","vezes","viagem","vindo","vinte","você","vocês","vos","vossa","vossas","vosso","vossos","vários","vão","vêm","vós","zero","à","às","área","é","éramos","és","último"
}

def remover_acentos(texto: str) -> str:
  
    try:
        s = str(texto)
    except Exception:
        s = '' if texto is None else str(texto)
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def simple_word_tokenize(texto: str) -> List[str]:
  
    s = remover_acentos(texto or '')
    token_re = re.compile(r"[A-Za-z_+#]+")
    toks = [t.lower() for t in token_re.findall(s) if len(t) > 1]
   
    try:
        norm_stop = {remover_acentos(w).lower() for w in STOPWORDS}
    except Exception:
        norm_stop = {w.lower() for w in STOPWORDS}

    cleaned: List[str] = []
    for t in toks:
        if any(ch.isdigit() for ch in t):
            continue
        if t in norm_stop:
            continue
        cleaned.append(t)
    return cleaned


def sanitize_for_json(obj: Any) -> Any:
   
    try:
        import pandas as _pd
    except Exception:
        _pd = None
    try:
        import numpy as _np
    except Exception:
        _np = None

    def _is_na(x):
        try:
            if _pd is not None:
                return bool(_pd.isna(x))
        except Exception:
            pass
        try:
            import math
            if isinstance(x, float) and math.isnan(x):
                return True
        except Exception:
            pass
        try:
            if _np is not None and isinstance(x, _np.floating):
                return bool(_np.isnan(x))
        except Exception:
            pass
        return False

    def _rec(v):
        # dict
        if isinstance(v, dict):
            return {k: _rec(val) for k, val in v.items()}
        # list/tuple
        if isinstance(v, (list, tuple)):
            return [_rec(i) for i in v]
        # pandas types
        if _pd is not None:
            try:
                if isinstance(v, _pd.Timestamp):
                    return v.isoformat()
                if isinstance(v, _pd.Timedelta):
                    return str(v)
                # convert pandas scalar
                if isinstance(v, _pd.Series) or isinstance(v, _pd.DataFrame):
                    # not expected here, stringify as repr
                    return repr(v)
            except Exception:
                pass
        # numpy scalar
        if _np is not None:
            try:
                if isinstance(v, _np.generic):
                    try:
                        return v.item()
                    except Exception:
                        return float(v)
            except Exception:
                pass
        # NA check
        try:
            if _is_na(v):
                return None
        except Exception:
            pass
        return v

    return _rec(obj)

TEXT_EXT = ('.csv', '.txt', '.json')


def build_user_token(result: Dict[str, Any]) -> Dict[str, List[str]]:
    """Gera coleções de tokens a partir do dicionário `result`.

    Para cada chave relevante em `result` (por exemplo, `df_skills`, `df_positions`,
    `df_projects`, `df_languages`, `df_profile`, `df_job_seeker`,
    `df_job_applicant_saved_answers`, `df_education`) extrai tokens dos campos
    de interesse (Title, Name, Headline, Answer, etc.), removendo dígitos e hifens,
    tokenizando com a heurística [A-Za-z+#]+, normalizando para lowercase,
    removendo tokens curtos e retornando conjuntos (listas ordenadas) por categoria.

    Retorna um dict com as coleções por categoria e uma chave adicional 'combined'
    com a união de todos os tokens.
    """
    if not isinstance(result, dict):
        return {}

    token_re = re.compile(r"[A-Za-z0-9_+#]+")

    category_fields = {
        'df_skills': ['name', 'title'],
        'df_positions': ['title', 'name'],
        'df_projects': ['title', 'name'],
        'df_languages': ['title', 'name'],
        'df_profile': ['headline', 'geo location', 'industry'],
        'df_job_seeker': ['job titles', 'locations'],
        'df_job_applicant_saved_answers': ['answer', 'question'],
        'df_education': ['degree name', 'school name'],
    }

    results: Dict[str, List[str]] = {}
    combined_tokens: set = set()

    for cat, fields in category_fields.items():
        records = result.get(cat) or []
        tokens_set = set()
        if not isinstance(records, list):
            results[cat] = []
            continue

        for rec in records:
            if not isinstance(rec, dict):
                continue
            mapping = {k.lower(): k for k in rec.keys()}
            for f in fields:
                actual_key = mapping.get(f)
                if actual_key is None:
                    for k_lower in mapping.keys():
                        if f in k_lower:
                            actual_key = mapping[k_lower]
                            break
                if actual_key is None:
                    continue
                val = rec.get(actual_key)
                if val is None:
                    continue
                try:
                    s = str(val)
                except Exception:
                    continue
                s_clean = re.sub(r"[-]+", " ", s)
                try:
                    toks = simple_word_tokenize(s_clean)
                except Exception:
                    toks = token_re.findall(s_clean)

                for m in toks:

                    if not isinstance(m, str):
                        continue
                    mm = token_re.findall(m)
                    for part in mm:
                        t = part.strip().lower()
                        if len(t) > 1:
                            tokens_set.add(t)
                            combined_tokens.add(t)

        results[cat] = sorted(tokens_set)

    results['combined'] = sorted(combined_tokens)
    return results


def jobspy_search(user_token: Any,
                  location: str = "São Paulo, SP",
                  search_term: str ='("Estágio em Dados" OR "Estágio em Engenheiro de Dados" OR "Programa de Estágio" OR "Estágio em Ciência de Dados" OR "Estágio em Analista de Dados")',
                  max_results: int = 10) -> List[Dict[str, Any]]:

    try:
        raw = scrape_jobs(
            site_name=["linkedin", "indeed"],
            search_term=search_term,
            location=location,
            results_wanted=max_results,
            linkedin_fetch_description=True,
            distance=100
        )
    except Exception:
        return []

    try:
        if 'pd' in globals() and pd is not None and isinstance(raw, pd.DataFrame):
            return raw.to_dict(orient='records')

        if isinstance(raw, list):
            normalized: List[Dict[str, Any]] = []
            for item in raw:
                if 'pd' in globals() and pd is not None and isinstance(item, pd.Series):
                    normalized.append(item.to_dict())
                elif isinstance(item, dict):
                    normalized.append(item)
                else:
                    try:
                        normalized.append(dict(item))
                    except Exception:
                        normalized.append({'value': item})
            return normalized

        if isinstance(raw, dict):
            return [raw]

        return [{'value': str(raw)}]
    except Exception:
        return []


def build_jobs_token(jobs: Any) -> Any:
    """Tokeniza o campo de descrição de cada item em `jobs` e adiciona
    a chave 'description_token' com a lista de tokens.

    Usa NLTK quando disponível e faz fallback para uma expressão regular.
    Modifica os itens in-place e também retorna a lista modificada.
    """
    token_re = re.compile(r"[A-Za-z0-9_+#]+")

    if not isinstance(jobs, list):
        return []

    for item in jobs:
        try:
            desc = None
            if isinstance(item, dict):
                for k in ('description', 'job_function', 'job_level', 'job_type', 'location', 'skills'):
                    if k in item and item[k]:
                        desc = item[k]
                        break
                if desc is None:
                    for v in item.values():
                        if isinstance(v, str) and len(v) > 30:
                            desc = v
                            break
                s = str(desc) if desc is not None else ''
            elif 'pd' in globals() and pd is not None and isinstance(item, pd.Series):
                s = ''
                for k in ('description', 'job_function', 'job_level', 'job_type', 'location', 'skills'):
                    if k in item.index:
                        s = str(item.get(k, ''))
                        break
            else:
                s = str(item)

            s_clean = re.sub(r"[-]+", " ", s)

            try:
                toks = simple_word_tokenize(s_clean)
            except Exception:
                toks = token_re.findall(s_clean)

            tokens_set = set()
            for m in toks:
                if not isinstance(m, str):
                    continue
                mm = token_re.findall(m)
                for part in mm:
                    t = part.strip().lower()
                    if len(t) > 1:
                        tokens_set.add(t)

            tokens_list = sorted(tokens_set)
            if isinstance(item, dict):
                item['description_token'] = tokens_list
        except Exception:
            try:
                if isinstance(item, dict):
                    item.setdefault('description_token', [])
            except Exception:
                pass

    return jobs


def add_adamic_adar_scores(user_token: Any, jobs: Any, token_field: str = 'description_token') -> Any:
   
    user_tokens = []
    if isinstance(user_token, dict):
        if isinstance(user_token.get('combined'), list):
            user_tokens = [t for t in user_token.get('combined') if isinstance(t, str)]
        else:
            for v in user_token.values():
                if isinstance(v, list):
                    user_tokens.extend([t for t in v if isinstance(t, str)])
    elif isinstance(user_token, list):
        user_tokens = [t for t in user_token if isinstance(t, str)]

    user_tokens_set = set(user_tokens)

    G = nx.Graph()
    user_node = 'user:0'
    G.add_node(user_node)

    for t in user_tokens_set:
        if not isinstance(t, str):
            continue
        tt = remover_acentos(t).lower()
        tok_node = f'tok:{tt}'
        if not G.has_node(tok_node):
            G.add_node(tok_node)
        G.add_edge(user_node, tok_node)

    job_nodes = []  
    for idx, job in enumerate(jobs):
        job_id = None
        if isinstance(job, dict):
            job_id = job.get('id') or job.get('job_url') or job.get('job_url_direct')
        if job_id is None:
            job_id = f'job:{idx}'
        job_node = f'job:{job_id}'
        G.add_node(job_node)
        job_nodes.append((job_node, idx))

        try:
            toks = []
            if isinstance(job, dict):
                toks = job.get(token_field) or []
            elif 'pd' in globals() and pd is not None and isinstance(job, pd.Series):
                toks = job.get(token_field, []) if token_field in job.index else []
            if isinstance(toks, str):
                toks = [toks]
            for t in toks:
                if t is None:
                    continue
                try:
                    if not isinstance(t, str):
                        t = str(t)
                except Exception:
                    continue
                tt = remover_acentos(t).lower()
                tok_node = f'tok:{tt}'
                if not G.has_node(tok_node):
                    G.add_node(tok_node)
                G.add_edge(job_node, tok_node)
        except Exception:
            continue


    pairs = [(user_node, jn) for jn, _ in job_nodes]
    scores = {jn: 0.0 for jn, _ in job_nodes}
    try:
        from networkx.algorithms.link_prediction import jaccard_coefficient, adamic_adar_index
        jac_iter = list(adamic_adar_index(G, pairs))
        for (_u, v, p) in jac_iter:
            scores[v] = float(p)
    except Exception:
        for jn, idx in job_nodes:
            try:
                user_n = set(G.neighbors(user_node))
                job_n = set(G.neighbors(jn))
                # token nodes use prefix 'tok:'
                user_tok = {n for n in user_n if n.startswith('tok:')}
                job_tok = {n for n in job_n if n.startswith('tok:')}
                inter = user_tok & job_tok
                union = user_tok | job_tok
                scores[jn] = float(score)
            except Exception:
                scores[jn] = 0.0

    for job_node, idx in job_nodes:
        score = scores.get(job_node, 0.0)
        try:
            jobs[idx]['adamic_adar'] = float(score)
        except Exception:
            try:
                jobs[idx]['adamic_adar'] = score
            except Exception:
                pass

    return jobs


def compute_idf_map(jobs: List[Dict[str, Any]], token_field: str = 'description_token') -> Dict[str, float]:

    import math
    N = 0
    df = {}
    if not isinstance(jobs, list):
        return {}
    for job in jobs:
        toks = []
        if isinstance(job, dict):
            toks = job.get(token_field) or []
        if isinstance(toks, str):
            toks = [toks]
        if not toks:
            continue
        N += 1
        seen = set()
        for t in toks:
            if not isinstance(t, str):
                continue
            tt = remover_acentos(t).lower()
            if tt in seen:
                continue
            seen.add(tt)
            df[tt] = df.get(tt, 0) + 1

    idf = {}
    for t, cnt in df.items():
        try:
            idf[t] = math.log((1.0 + max(1, N)) / (1.0 + cnt)) + 1.0
        except Exception:
            idf[t] = 1.0
    return idf


def weighted_adamic_adar(user_token: Any, jobs: Any, idf_map: Optional[Dict[str, float]] = None, token_field: str = 'description_token') -> Any:
    """Compute a weighted Adamic–Adar style score using idf_map.

    Adds field 'adamic_weighted' to each job dict and returns jobs list.
    """
    import math
    try:
        import networkx as _nx
    except Exception:
        _nx = None

    if idf_map is None:
        idf_map = {}

    user_tokens = []
    if isinstance(user_token, dict):
        user_tokens = [t for t in user_token.get('combined', []) if isinstance(t, str)]
    elif isinstance(user_token, list):
        user_tokens = [t for t in user_token if isinstance(t, str)]
    user_tokens_set = {remover_acentos(t).lower() for t in user_tokens}


    use_graph = _nx is not None
    G = _nx.Graph() if use_graph else None
    user_node = 'user:0'
    if use_graph:
        G.add_node(user_node)
        for t in user_tokens_set:
            G.add_node(f'tok:{t}')
            G.add_edge(user_node, f'tok:{t}')

    job_nodes = []
    for idx, job in enumerate(jobs if isinstance(jobs, list) else []):
        toks = []
        if isinstance(job, dict):
            toks = job.get(token_field) or []
        if isinstance(toks, str):
            toks = [toks]
        norm_toks = [remover_acentos(t).lower() for t in toks if isinstance(t, str)]
        if use_graph:
            jn = f'job:{idx}'
            G.add_node(jn)
            for t in norm_toks:
                tn = f'tok:{t}'
                if not G.has_node(tn):
                    G.add_node(tn)
                G.add_edge(jn, tn)
        job_nodes.append((idx, norm_toks))

    for idx, toks in job_nodes:
        sc = 0.0
        for t in set(toks) & user_tokens_set:
            deg = None
            if use_graph:
                try:
                    deg = max(1, G.degree(f'tok:{t}'))
                except Exception:
                    deg = 1
            else:
                deg = max(1, 1 + (0 if idf_map is None else 0))
            idf = float(idf_map.get(t, 1.0)) if idf_map is not None else 1.0
            try:
                sc += (idf / max(1.0, math.log(1.0 + deg)))
            except Exception:
                sc += idf
        try:
            if isinstance(jobs[idx], dict):
                jobs[idx]['adamic_weighted'] = float(sc)
        except Exception:
            pass
    return jobs


def tfidf_cosine_scores(user_token: Any, jobs: Any, token_field: str = 'description_token') -> List[float]:
    """Compute TF-IDF cosine similarity between user combined tokens and each job description tokens.

    Falls back to simple token-overlap ratio if scikit-learn is unavailable.
    Returns list of floats (0..1) aligned with jobs list.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        TfidfVectorizer = None

    
    if isinstance(user_token, dict):
        user_doc = ' '.join([t for t in user_token.get('combined', []) if isinstance(t, str)])
    elif isinstance(user_token, list):
        user_doc = ' '.join([t for t in user_token if isinstance(t, str)])
    else:
        user_doc = ''

    jobs_list = jobs if isinstance(jobs, list) else []
    docs = []
    for job in jobs_list:
        toks = []
        if isinstance(job, dict):
            toks = job.get(token_field) or []
        if isinstance(toks, str):
            toks = [toks]
        docs.append(' '.join([remover_acentos(t).lower() for t in toks if isinstance(t, str)]))

    if TfidfVectorizer is None or not user_doc or not any(docs):
        
        out = []
        user_set = set([remover_acentos(t).lower() for t in (user_doc or '').split() if t])
        for d in docs:
            jset = set([w for w in d.split() if w])
            inter = user_set & jset
            union = user_set | jset
            score = (len(inter) / len(union)) if union else 0.0
            out.append(float(score))
        return out

    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        all_docs = [user_doc] + docs
        X = vec.fit_transform(all_docs)
        sims = cosine_similarity(X[0:1], X[1:]).flatten()
        return [float(s) for s in sims]
    except Exception:
        
        out = []
        user_set = set([remover_acentos(t).lower() for t in (user_doc or '').split() if t])
        for d in docs:
            jset = set([w for w in d.split() if w])
            inter = user_set & jset
            union = user_set | jset
            score = (len(inter) / len(union)) if union else 0.0
            out.append(float(score))
        return out


def _title_stage_multiplier(title: str) -> float:
    """If title indicates an 'estágio', apply role-based multiplier."""
    if not title or not isinstance(title, str):
        return 1.0
    t = remover_acentos(title).lower()
    if 'estagi' not in t and 'estágio' not in t and 'estagio' not in t and 'intern' not in t:
        return 1.0
    
    mapping = {
        'cientista': 1.2,
        'engenheiro': 1.3,
        'analista': 1.1,
        'dados': 1.3,
    }
    for k, v in mapping.items():
        if k in t:
            return float(v)
    return 1.15


def jobs_pyvis_html(user_token: Any, jobs: Any, token_field: str = 'description_token',
                    max_tokens: int = 200, height: str = '100vh', width: str = '100%') -> str:
    """Build a pyvis HTML visualization for the bipartite token-job graph.

    Returns an HTML string suitable for embedding inside an <iframe>.
    If pyvis is not installed or an error occurs, returns an empty string.
    """
    try:
        from pyvis.network import Network
        have_pyvis = True
    except Exception:
        Network = None
        have_pyvis = False

    
    token_counts = Counter()
    user_tokens = set()
    if isinstance(user_token, dict):
        for v in user_token.get('combined', []):
            if isinstance(v, str):
                token_counts[v] += 1
                user_tokens.add(v)
    elif isinstance(user_token, list):
        for v in user_token:
            if isinstance(v, str):
                token_counts[v] += 1
                user_tokens.add(v)

    jobs_list = jobs if isinstance(jobs, list) else []
    for job in jobs_list:
        toks = []
        if isinstance(job, dict):
            toks = job.get(token_field) or []
        elif 'pd' in globals() and pd is not None and isinstance(job, pd.Series):
            toks = job.get(token_field, []) if token_field in job.index else []
        if isinstance(toks, str):
            toks = [toks]
        for t in toks:
            if isinstance(t, str):
                token_counts[t] += 1

    if not token_counts:
        return ''

    
    most_common = [t for t, _ in token_counts.most_common(max_tokens)]
    token_set = set(most_common) | user_tokens

    
    nodes = []
    edges = []

    user_node = 'user:0'
    nodes.append({'id': user_node, 'label': 'User', 'color': '#2b83ba', 'title': 'User tokens'})

    for t in sorted(token_set):
        nid = f'tok:{t}'
        nodes.append({'id': nid, 'label': t, 'color': '#7fc97f', 'title': f'Token: {t}', 'value': 1})
        if t in user_tokens:
            edges.append({'from': user_node, 'to': nid})

    for idx, job in enumerate(jobs_list):
        job_id = None
        if isinstance(job, dict):
            job_id = job.get('id') or job.get('job_url') or job.get('job_url_direct')
        if job_id is None:
            job_id = f'job:{idx}'
        job_node = f'job:{job_id}'
        title = ''
        score = 0.0
        if isinstance(job, dict):
            title = job.get('title') or str(job_id)
            try:
                score = float(job.get('adamic_adar', 0.0) or 0.0)
            except Exception:
                score = 0.0
        size = 10 + min(40, int(score * 40))
        nodes.append({'id': job_node, 'label': title, 'color': '#fb8072', 'title': str(title), 'value': size})

        toks = []
        if isinstance(job, dict):
            toks = job.get(token_field) or []
        elif 'pd' in globals() and pd is not None and isinstance(job, pd.Series):
            toks = job.get(token_field, []) if token_field in job.index else []
        if isinstance(toks, str):
            toks = [toks]
        for t in toks:
            if not isinstance(t, str):
                continue
            nid = f'tok:{t}'
            if t in token_set:
                edges.append({'from': job_node, 'to': nid})

    if have_pyvis:
        try:
            net = Network(height=height, width=width, notebook=False)
            net.toggle_physics(False)
            net.neighborhood_highlight(True)
            net.show_buttons(filter="physics")
            for n in nodes:
                net.add_node(n['id'], label=n.get('label'), color=n.get('color'), title=n.get('title'), size=n.get('value', 8))
            for e in edges:
                net.add_edge(e['from'], e['to'])
            try:
                return net.generate_html()
            except Exception:
                import tempfile
                tf = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
                path = tf.name
                tf.close()
                net.save_graph(path)
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception:
            pass

    try:
        nodes_json = json.dumps(nodes, ensure_ascii=False)
        edges_json = json.dumps(edges, ensure_ascii=False)
        html = f"""<!doctype html>
<html>
  <head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <link href='https://unpkg.com/vis-network@9.1.2/styles/vis-network.css' rel='stylesheet' type='text/css' />
    <style>html,body{{margin:0;padding:0;height:100%;}}#mynetwork{{width:100%;height:100%;}}</style>
  </head>
  <body>
    <div id='mynetwork'></div>
    <script src='https://unpkg.com/vis-network@9.1.2/standalone/umd/vis-network.min.js'></script>
        <script>
            const nodes = new vis.DataSet({nodes_json});
            const edges = new vis.DataSet({edges_json});
      const container = document.getElementById('mynetwork');
      const data = {{ nodes: nodes, edges: edges }};
      const options = {{ physics: {{ stabilization: false }}, nodes: {{ shape: 'dot', font: {{multi:true}} }} }};
      const network = new vis.Network(container, data, options);
    </script>
  </body>
</html>"""
        return html
    except Exception:
        return ''

def extract_zip_file_bytes(zip_bytes: bytes) -> Dict[str, bytes]:
    """Extrai um zip a partir de bytes e retorna um dict filename -> bytes.

    Mantém todos os arquivos em memória (não grava em disco).
    """
    # delega para a instância padrão (LinkedInProcessor)
    return linkedin_processor.extract_zip_file_bytes(zip_bytes)

def read_file_text(data: bytes, filename: str) -> str:
    """Tenta decodificar bytes para texto e, quando for CSV, retorna o conteúdo como string.

    Não faz parsing profundo aqui — isso fica para a análise.
    """
    return linkedin_processor.read_file_text(data, filename)

def process_zip_and_combine(zip_bytes: bytes) -> Dict[str, Any]:
    """Extrai o zip em memória, seleciona todos os CSVs relevantes e combina em um DataFrame.

    Retorna um dicionário com chaves:
      - combined_df: o DataFrame (ou None se pandas não disponível)
      - combined_preview: primeiros registros como lista de dicts
      - files_seen: lista de arquivos CSV considerados
    """
    return linkedin_processor.process_zip_and_combine(zip_bytes)


class LinkedInProcessor:
    """Classe encapsula o processamento de um export LinkedIn ZIP.

    Uso:
        proc = LinkedInProcessor()
        result = proc.process_zip_and_combine(zip_bytes)
    """

    def __init__(self, keywords: Optional[List[str]] = None, max_files: int = 4):
        self.max_files = max_files
        if keywords is None:
            self.keywords = ['projects', 'profile', 'skills', 'languages',
                             'job applicant saved answers', 'education',
                             'job seeker', 'positions']
        else:
            self.keywords = keywords

    def extract_zip_file_bytes(self, zip_bytes: bytes) -> Dict[str, bytes]:
        files: Dict[str, bytes] = {}
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for zi in z.infolist():
                if zi.is_dir():
                    continue
                name = zi.filename
                try:
                    data = z.read(zi)
                except Exception:
                    continue
                files[name] = data
        return files

    def read_file_text(self, data: bytes, filename: str) -> str:
        try:
            text = data.decode('utf-8')
        except Exception:
            try:
                text = data.decode('latin-1')
            except Exception:
                text = data.decode('utf-8', errors='replace')
        return text

    def process_zip_and_combine(self, zip_bytes: bytes) -> Dict[str, Any]:
        files = self.extract_zip_file_bytes(zip_bytes)
        KEYWORDS = self.keywords

        csv_texts: Dict[str, str] = {}
        files_seen: List[str] = []
        for name, data in files.items():
            lname = name.lower()
            if not lname.endswith('.csv'):
                continue
            try:
                text = self.read_file_text(data, name)
            except Exception:
                continue
            csv_texts[name] = text
            files_seen.append(name)

        result: Dict[str, Any] = {}

        for kw in KEYWORDS:
            safe_key = 'df_' + re.sub(r'[^0-9a-zA-Z]+', '_', kw).strip('_')
            result[safe_key] = []

            # find files matching this keyword
            for fname, text in csv_texts.items():
                if kw in fname.lower():
                    try:
                        if pd is not None:
                            df = pd.read_csv(io.StringIO(text))
                            records = df.to_dict(orient='records')
                        else:
                            reader = csv.DictReader(io.StringIO(text))
                            records = [r for r in reader]
                    except Exception:
                        records = []

                    # If this is the education keyword, keep only Degree Name and School Name
                    if kw.strip().lower() == 'education' and records:
                        filtered: List[Dict[str, Any]] = []
                        for r in records:
                            # match keys ignoring case
                            mapping = {k.lower(): k for k in r.keys()}
                            row: Dict[str, Any] = {}
                            for desired in ('Degree Name', 'School Name'):
                                key = mapping.get(desired.lower())
                                row[desired] = r.get(key) if key is not None else None
                            filtered.append(row)
                        result[safe_key].extend(filtered)
                    # If this is the job seeker keyword, keep only Job Titles and Locations
                    elif kw.strip().lower() == 'job seeker' and records:
                        filtered: List[Dict[str, Any]] = []
                        for r in records:
                            mapping = {k.lower(): k for k in r.keys()}
                            row: Dict[str, Any] = {}
                            for desired in ('Job Titles', 'Locations'):
                                key = mapping.get(desired.lower())
                                row[desired] = r.get(key) if key is not None else None
                            filtered.append(row)
                        result[safe_key].extend(filtered)
                    # If this is the languages keyword, keep only Title
                    elif kw.strip().lower() == 'languages' and records:
                        filtered: List[Dict[str, Any]] = []
                        for r in records:
                            mapping = {k.lower(): k for k in r.keys()}
                            row: Dict[str, Any] = {}
                            # some exports use 'title' or 'name'
                            key = mapping.get('title') or mapping.get('name')
                            row['Title'] = r.get(key) if key is not None else None
                            filtered.append(row)
                        result[safe_key].extend(filtered)
                    # If this is the positions keyword, keep only Title
                    elif kw.strip().lower() == 'positions' and records:
                        filtered: List[Dict[str, Any]] = []
                        for r in records:
                            mapping = {k.lower(): k for k in r.keys()}
                            row: Dict[str, Any] = {}
                            key = mapping.get('title') or mapping.get('name')
                            row['Title'] = r.get(key) if key is not None else None
                            filtered.append(row)
                        result[safe_key].extend(filtered)
                    # If this is the profile keyword, keep only Headline, Geo Location and Industry
                    elif kw.strip().lower() == 'profile' and records:
                        filtered: List[Dict[str, Any]] = []
                        for r in records:
                            mapping = {k.lower(): k for k in r.keys()}
                            row: Dict[str, Any] = {}
                            for desired in ('Headline', 'Geo Location', 'Industry'):
                                key = mapping.get(desired.lower())
                                row[desired] = r.get(key) if key is not None else None
                            filtered.append(row)
                        result[safe_key].extend(filtered)
                    # If this is the projects keyword, keep only Title
                    elif kw.strip().lower() == 'projects' and records:
                        filtered: List[Dict[str, Any]] = []
                        for r in records:
                            mapping = {k.lower(): k for k in r.keys()}
                            row: Dict[str, Any] = {}
                            key = mapping.get('title') or mapping.get('name')
                            row['Title'] = r.get(key) if key is not None else None
                            filtered.append(row)
                        result[safe_key].extend(filtered)
                    else:
                        result[safe_key].extend(records)
        
        try:
            user_token = build_user_token(result)
            jobs_search = jobspy_search(user_token)
            jobs_token = build_jobs_token(jobs_search)

            
            unique_jobs: List[Dict[str, Any]] = []
            unique_tokens: List[Dict[str, Any]] = []
            seen_keys = set()
            for idx, job in enumerate(jobs_search if isinstance(jobs_search, list) else []):
                try:
                    if isinstance(job, dict):
                        key = job.get('id') or job.get('job_url') or job.get('job_url_direct')
                        if not key:
                            
                            title = (job.get('title') or job.get('position') or job.get('job_title') or '')
                            company = (job.get('company') or job.get('company_name') or '')
                            key = f"{title}||{company}"
                    else:
                        
                        key = str(job)
                except Exception:
                    key = str(idx)

                if key in seen_keys:
                    
                    continue
                seen_keys.add(key)
                unique_jobs.append(job)
                
                try:
                    tok_entry = jobs_token[idx] if isinstance(jobs_token, list) and idx < len(jobs_token) else None
                except Exception:
                    tok_entry = None
                if tok_entry is not None:
                    unique_tokens.append(tok_entry)
                else:
                    unique_tokens.append({})

            
            jobs_search = unique_jobs
            jobs_token = unique_tokens

            
            jobs_with_scores = add_adamic_adar_scores(user_token, jobs_token)

            
            try:
                idf_map = compute_idf_map(jobs_with_scores, token_field='description_token')
            except Exception:
                idf_map = {}

            
            try:
                jobs_with_scores = weighted_adamic_adar(user_token, jobs_with_scores, idf_map=idf_map, token_field='description_token')
            except Exception:
                pass

            
            try:
                tfidf_scores = tfidf_cosine_scores(user_token, jobs_with_scores, token_field='description_token')
            except Exception:
                tfidf_scores = [0.0 for _ in (jobs_with_scores if isinstance(jobs_with_scores, list) else [])]

            
            try:
                aw_vals = [float(j.get('adamic_weighted') or 0.0) for j in (jobs_with_scores if isinstance(jobs_with_scores, list) else [])]
                max_aw = max(aw_vals) if aw_vals else 0.0
                norm_aw = [(v / max_aw) if max_aw and max_aw > 0 else 0.0 for v in aw_vals]
            except Exception:
                norm_aw = [0.0 for _ in (jobs_with_scores if isinstance(jobs_with_scores, list) else [])]

            
            try:
                if isinstance(user_token, dict):
                    user_tokens = [remover_acentos(t).lower() for t in user_token.get('combined', []) if isinstance(t, str)]
                elif isinstance(user_token, list):
                    user_tokens = [remover_acentos(t).lower() for t in user_token if isinstance(t, str)]
                else:
                    user_tokens = []
                user_tokens_set = set(user_tokens)
            except Exception:
                user_tokens_set = set()

            
            
            w_cov = 0.5
            w_adamic = 0.3
            w_tfidf = 0.2

            combined_raw = []
            for i, job in enumerate(jobs_with_scores if isinstance(jobs_with_scores, list) else []):
                
                toks = []
                try:
                    if isinstance(job, dict):
                        toks = job.get('description_token') or job.get('description_tokens') or []
                    if isinstance(toks, str):
                        toks = [toks]
                    norm_job_toks = [remover_acentos(t).lower() for t in toks if isinstance(t, str)]
                except Exception:
                    norm_job_toks = []

                
                try:
                    if norm_job_toks:
                        covered = len(set(norm_job_toks) & user_tokens_set)
                        coverage = float(covered) / float(len(set(norm_job_toks)))
                    else:
                        coverage = 0.0
                except Exception:
                    coverage = 0.0
                try:
                    job['coverage'] = float(coverage)
                except Exception:
                    pass

                a = norm_aw[i] if i < len(norm_aw) else 0.0
                t = float(tfidf_scores[i]) if i < len(tfidf_scores) else 0.0

                base = (w_cov * coverage) + (w_adamic * a) + (w_tfidf * t)

                
                try:
                    title = job.get('title') or job.get('position') or ''
                    mult = _title_stage_multiplier(title)
                    base = base * float(mult)
                except Exception:
                    pass

                combined_raw.append(float(base))

            
            try:
                max_comb = max(combined_raw) if combined_raw else 0.0
                if max_comb and max_comb > 0:
                    combined_norm = [c / max_comb for c in combined_raw]
                else:
                    combined_norm = [0.0 for _ in combined_raw]
            except Exception:
                combined_norm = [0.0 for _ in combined_raw]

            
            try:
                for i, job in enumerate(jobs_with_scores if isinstance(jobs_with_scores, list) else []):
                    sc = float(combined_norm[i]) if i < len(combined_norm) else 0.0
                    try:
                        job['adamic_adar'] = float(sc)
                    except Exception:
                        job['adamic_adar'] = sc
                    try:
                        job['score_percent'] = int(round(sc * 100))
                    except Exception:
                        job['score_percent'] = int(sc * 100) if isinstance(sc, (int, float)) else 0
                    
                    try:
                        job['tfidf_score'] = float(tfidf_scores[i]) if i < len(tfidf_scores) else 0.0
                    except Exception:
                        pass
                    try:
                        job['adamic_weighted_raw'] = float(job.get('adamic_weighted') or 0.0)
                    except Exception:
                        pass
            except Exception:
                pass

            
            try:
                if isinstance(jobs_with_scores, list):
                    jobs_with_scores.sort(key=lambda jj: float(jj.get('adamic_adar') or 0.0), reverse=True)
            except Exception:
                pass

            
            def sanitize_for_json(x: Any) -> Any:
                
                try:
                    if 'pd' in globals() and pd is not None and pd.isna(x):
                        return None
                except Exception:
                    pass
                
                try:
                    if isinstance(x, float) and math.isnan(x):
                        return None
                except Exception:
                    pass
                
                if isinstance(x, (bytes, bytearray)):
                    try:
                        return x.decode('utf-8')
                    except Exception:
                        return str(x)
                
                if isinstance(x, dict):
                    return {k: sanitize_for_json(v) for k, v in x.items()}
                
                if isinstance(x, list):
                    return [sanitize_for_json(v) for v in x]
                if isinstance(x, tuple):
                    return tuple(sanitize_for_json(v) for v in x)
                
                try:
                    if hasattr(x, 'item'):
                        try:
                            val = x.item()
                            if val is x:
                                return x
                            return sanitize_for_json(val)
                        except Exception:
                            pass
                except Exception:
                    pass
                return x

            
            graph_url = None
            try:
                html = jobs_pyvis_html(user_token, jobs_with_scores)
                if html and isinstance(html, str) and html.strip():
                    
                    root = Path(__file__).resolve().parents[1]
                    graphs_dir = root / 'temp_graphs'
                    try:
                        graphs_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    gid = uuid.uuid4().hex
                    fname = graphs_dir / f'{gid}.html'
                    try:
                        with open(fname, 'w', encoding='utf-8') as f:
                            f.write(html)
                        graph_url = f'/graph/{gid}'
                    except Exception:
                        graph_url = None
            except Exception:
                graph_url = None

            
            payload: Dict[str, Any] = {}
            payload['user_token'] = sanitize_for_json(user_token)
            payload['jobs_token'] = sanitize_for_json(jobs_with_scores)
            if graph_url:
                payload['jobs_graph_url'] = graph_url
        except Exception:
            
            payload = payload if 'payload' in locals() else {}
            payload['user_token'] = {}
            payload['jobs_token'] = []

        
        try:
            payload = sanitize_for_json(payload)
        except Exception:
            pass
        return payload

    def handle_flask_request_file(self, flask_request) -> Tuple[Dict[str, Any], int]:
        upload = None
        if 'file' in flask_request.files:
            upload = flask_request.files['file']
            data = upload.read()
        else:
            data = flask_request.get_data() or b''

        if not data:
            return ({'error': 'Nenhum arquivo recebido'}, 400)

        try:
            result = self.process_zip_and_combine(data)
            return (result, 200)
        except zipfile.BadZipFile:
            return ({'error': 'Arquivo zip inválido'}, 400)
        except Exception as e:
            return ({'error': f'Erro interno: {e}'}, 500)


linkedin_processor = LinkedInProcessor()

def handle_flask_request_file(flask_request) -> Tuple[Dict[str, Any], int]:

    return linkedin_processor.handle_flask_request_file(flask_request)
