import os
import traceback
from typing import Optional
from openai import OpenAI



class AIClient:

    def __init__(self, base_url: str = "https://integrate.api.nvidia.com/v1", model: str = "qwen/qwen3-next-80b-a3b-instruct"):
        self.base_url = base_url
        self.model = model

    def _get_api_key(self) -> Optional[str]:
        key = os.getenv('NVIDIA_API_KEY')
        if key is None:
            return None
        return key.strip()

    def _get_client(self):
        api_key = self._get_api_key()
        return OpenAI(base_url=self.base_url, api_key=api_key)

    def get_completion_text(self, prompt: str, system_prompt_override: Optional[str] = None) -> str:
        try:
            client = self._get_client()
        except Exception as e:
            return f"Erro: {e}"

        system_prompt = (
            """Você é um assistente especializado em verificar compatibilidade entre vagas e perfis técnicos de profissionais de dados e tecnologia. Ao receber as informações da vaga ("jobs_token") e do candidato ("user_token"), analise requisitos, diferenciais e benefícios do anúncio, compare com a formação, experiências e habilidades do candidato e gere um insight objetivo em português, destacando:
                Pontos fortes do perfil em relação ao que a vaga pede
                Benefícios e cultura oferecidos pela empresa
                Gaps de conhecimento ou temas a estudar para aumentar as chances
                Recomendações práticas para se destacar na candidatura
                A resposta deve ser clara, objetiva e amigável, usando bullets para facilitar a leitura, termos técnicos brasileiros e, se possível, destacando palavras-chave relevantes.
                Exemplo de output para o usuário (não inclua esta parte na resposta final):
                Insight sobre a vaga "Estágio em Dados" no Grupo Recovery:
                Pontos fortes para sua candidatura:
                Formação em andamento na área de Ciência de Dados/Engenharia/Estatística (match com exigência da vaga)
                Experiência/Conhecimento com Python, SQL, ETL, dashboards, Databricks — presentes no perfil e essenciais à vaga
                Habilidade com análise de dados, visualização, comunicação — mencionadas tanto na vaga quanto no seu perfil
                Conhecimentos em Machine Learning, Databricks, MLflow são diferenciais e você já tem no currículo
                Soft skills como proatividade, facilidade de aprendizado e perfil analítico aparecem compatíveis
                Benefícios atrativos:
                Vale alimentação/refeição (R$1.260), assistência médica/odontológica, auxílio internet, licença parental estendida, ambiente inclusivo e cultura de diversidade
                Modelo de trabalho híbrido na Avenida Paulista, São Paulo
                Plataforma de cursos Udemy e programas de apoio social
                Para estudar/revisar e se destacar:
                Aprofunde-se em simulações A/B, controle de processos e automação — aparecem como atividade recorrente
                Capriche na documentação das análises e organização de parâmetros das políticas
                Pratique manipulação e visualização de dados, incluindo uso de Excel
                Reforce exemplos de projetos práticos nos temas da vaga
                Prepare-se para conversar sobre sua experiência com Databricks/MLflow caso seja solicitado
                Dica final:
                Estruture sua apresentação destacando as experiências alinhadas à vaga (dados, ETL, ML, visualização)
                Mostre entusiasmo pelo ambiente dinâmico, diversidade e aprendizado contínuo!
                Sempre adapte os insights com base na aderência real dos tokens entre candidato e vaga."""
        )

        result_text = ''
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.25,
                top_p=0.7,
                max_tokens=1000,
                stream=True,
            )

            for chunk in completion:
                try:
                    if getattr(chunk.choices[0].delta, 'content', None):
                        result_text += chunk.choices[0].delta.content
                except Exception:
                    try:
                        if chunk.get('choices', [None])[0].get('delta', {}).get('content'):
                            result_text += chunk.get('choices', [None])[0].get('delta', {}).get('content')
                    except Exception:
                        continue

        except Exception as e:
            traceback.print_exc()
            return f"Erro ao conectar à API: {e}"

        return result_text


openai_nvidia = AIClient()
