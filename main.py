import sys
from flask import Flask, render_template, request, jsonify
from flask import send_from_directory
import os
import json
import traceback
from pathlib import Path
from dotenv import load_dotenv
from utils.openai import openai_nvidia
from utils.linkedin import handle_flask_request_file
from flask import send_file, abort
from pathlib import Path
import os

env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(force=True)
    client = openai_nvidia

    if isinstance(data, dict) and 'user_token' in data and 'job' in data:
        try:
            user_token = data.get('user_token') or {}
            job = data.get('job') or {}
            score = job.get('adamic_adar') or job.get('score') or job.get('similarity') or 0
            desc = job.get('description') or job.get('job_description') or job.get('description_text') or job.get('description_token') or ''
            if isinstance(desc, list):
                desc = ' '.join([str(x) for x in desc])
            if not isinstance(desc, str):
                desc = str(desc)

            try:
                user_summary = ''
                if isinstance(user_token, dict):
                    combined = user_token.get('combined') if isinstance(user_token.get('combined'), list) else None
                    if combined:
                        user_summary = ' '.join(combined[:200])
                    else:
                        user_summary = ', '.join([k for k in user_token.keys()])
                else:
                    user_summary = str(user_token)
            except Exception:
                user_summary = ''

            prompt = (
                "Você é um assistente especialista em compatibilidade entre vagas e perfis técnicos de dados/tech. "
                "Com base nos tokens/resumo do candidato e na descrição da vaga, gere uma análise objetiva e prática. \n"
                "RETORNE APENAS JSON VÁLIDO com este esquema: {\n"
                "  \"strengths\": [string],\n"
                "  \"benefits\": [string] OR \"Não informado\",\n"
                "  \"gaps\": [string],\n"
                "  \"recommendations\": [string],\n"
                "  \"summary\": string\n"
                "}\n"
                "Cada array deve conter no máximo 6 items. Não inclua texto fora do JSON. Responda em português.\n\n"
                f"Perfil (tokens do usuário): {user_summary}\n\n"
                f"Vaga (trecho / descrição): {desc}\n\n"
                f"Score calculado: {score}\n\n"
            )

            try:
                response_text = client.get_completion_text(prompt)
            except Exception as e:
                tb = traceback.format_exc()
                return jsonify({'error': 'LLM request failed', 'detail': str(e), 'trace': tb}), 500

            try:
                parsed = json.loads(response_text)
                return jsonify({'analysis': parsed})
            except Exception:
                return jsonify({'analysis_text': response_text})
        except Exception as e:
            tb = traceback.format_exc()
            return jsonify({'error': 'Internal server error in /api/chat', 'detail': str(e), 'trace': tb}), 500


    raw = data.get('text') if 'text' in data else data.get('message')
    text = raw.strip() if isinstance(raw, str) else ''

    if not text:
        return jsonify({'error': 'Campo "text" (ou "message") é obrigatório.'}), 400

    prompt = (
        "Considere o texto abaixo e gere uma conclusão sucinta, clara e objetiva em linguagem formal:\n\n"
        f"{text}\n\n"
        "Conclusão:"
    )

    response_text = client.get_completion_text(prompt)
    return jsonify({'conclusion': response_text})

@app.route('/upload_linkedin_zip', methods=['POST'])
def upload_linkedin_zip():
    payload, status = handle_flask_request_file(request)
    return jsonify(payload), status


@app.route('/graph/<graph_id>')
def serve_graph(graph_id: str):
    root = Path(__file__).resolve().parents[0]
    graphs_dir = root / 'temp_graphs'
    fpath = graphs_dir / f"{graph_id}.html"
    if not fpath.exists():
        return abort(404)
    try:
        return send_file(str(fpath), mimetype='text/html')
    except Exception:
        return abort(500)


@app.route('/assets/<path:filename>')
def serve_assets(filename: str):
    root = Path(__file__).resolve().parents[0] / 'templates' / 'assets'
    fpath = root / filename
    if not fpath.exists():
        return abort(404)
    try:
        return send_from_directory(str(root), filename)
    except Exception:
        return abort(500)



if __name__ == '__main__':
    port = int(os.getenv('PORT', 5003))
    print(f"Server rodando na porta {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
