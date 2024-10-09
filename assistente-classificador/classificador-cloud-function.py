import time
import json
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

def generate(data_request):
    # Iniciando contagem do tempo de duração da automação
    start_time = time.time()
    conversa_json = data_request.get_json()["chat"]

    conversa_text = conversa_json.format()

    prompt = f'''
  **Tarefas:**
- Você deve identificar o principal (ou principais) motivo(s) de um cliente ter iniciado uma conversa com um atendente e atribuir esse principal (ou principais) motivo(s) de contato às classificações padronizadas já existentes em uma lista de classificações. Se identificar mais de um motivo de contato, pode utilizar mais de uma classificação padronizada. **Importante**: O máximo de classificações padronizadas para cada transcrição será de 10 classificações.
- Nessa lista, que estará indicada nesse prompt, você deve ler todas as linhas, que contém a classificação e sua explicação de quando utilizar ela, e identificar quais classificações se assemelham mais ao principal (ou principais) motivo(s) de contato entre cliente e atendente.
- Utilize a classificação "CONEXÃO PERDIDA" caso o cliente NÃO tenha mais que 2 interações ao longo da conversa.

**Lista de classificações que deve se basear para classificar a conversa:**
- VIAGENS AGENDADAS: Quando o cliente tiver alguma qualquer dúvida **(exceto dúvidas de custos, formas de pagamento ou estornos)** sobre um pacote de viagem ou viagem que fechou conosco, como dúvidas sobre remarcação de data da viagem, roteiro da viagem, passagens aéreas, hotéis reservados. Além de solicitações de confirmação de efetivação de alguma viagem agendada.

- FINANCEIRO: Utilizar quando cliente tiver dúvidas sobre como realizar pagamentos pela plataforma, quais são os tipos de pagamentos disponíveis. Também utilizar quando o cliente questionar se existem descontos para algum produto ou serviço, além de dúvidas sobre estornos de valores.

- CUSTOS: Utilizar quando o cliente informar alguma dúvida sobre qualquer custo que mencionar, como dúvida sobre um custo de um serviço ou produto, além de custos de impostos. Também utilizar caso o atendente informe que determinado valor que o cliente mencionar é referente a algum custo, seja custo de serviço, produto, entre outros e/ou custo de impostos.

- EXPERIÊNCIA DO USUÁRIO: Utilizar quando o cliente demonstrar alguma insatisfação com a plataforma de site e/ou aplicativo, também quando sugerir alguma melhoria na plataforma de site/e ou aplicativo ou informar uma instabilidade que identificou na plataforma de site e/ou aplicativo. Além de utilizar também caso o cliente informe alguma experiência negativa em um atendimento e/ou com um produto e/ou serviço.

**Como response você deve trazer apenas as seguintes informações:**
- Trazer apenas a classificação padronizada (ou as classificações padronizadas) do principal motivo (ou dos principais motivos) de contato da conversa entre cliente e atendente, tendo como base para a escolha da classificação (ou das classificações) a descrição de sua utilização. 

**Segue abaixo o padrão do output que você deve gerar como response:**

**Padrão do Output:**

Classificações: Classificação (ou classificações) padronizada(s) escolhida(s)

**Exemplos:**

**Input Exemplo:**

Cliente: Olá, estou com dúvidas sobre minha viagem marcada para o fim do mês com vocês.

Atendente: Olá! Certo, qual sua dúvida, por favor?

Cliente: Está tudo certo com a programação?

Atendente: Vou verificar em nosso sistema, poderia informar o número de pedido de sua viagem, por favor?

Cliente: Claro! É 123456

Atendente: Verifiquei em nosso sistema e está tudo certo!

Cliente: Perfeito, obrigado!

**Output Exemplo:**

Classificações: VIAGENS AGENDADAS

**Input Exemplo:**

Cliente: Olá, estou querendo fechar um pacote de viagem para Cancún com vocês, mas gostaria de saber quais são a formas de pagamento?

Atendente: Olá! Certo, nós aceitamos à vista via cartão de débito ou transferência bancária e também parcelado em até 12 vezes sem juros com cartão de crédito.

Cliente: Entendi, e vocês possuem algum desconto realizando o pagamento à vista?

Atendente: Sim, nesse caso você tem 5% de desconto na compra.

Cliente: Perfeito, obrigado!

Atendente: Por nada! Ajudo em algo mais?

Cliente: Não, somente isso.

**Output Exemplo:**

Classificações: FINANCEIRO

**Input Exemplo:**

Cliente: Olá, estava finalizando minha compra de uma viagem no site de vocês, mas vi que ao final houve um aumento do valor, qual o motivo do aumento?

Atendente: Olá! Nesse caso o aumento é devido a taxa de serviço da agência e aos impostos que recaem sobre a compra.

Cliente: Entendi, agora que vi aqui na plataforma de vocês essa informação, mas essa informação parece estar muito escondida no site. Seria ideal melhorar essa questão no site de vocês...

Atendente: Compreendo, pedimos desculpas pelo transtorno com essa experiência em nossa plataforma!

Cliente: Tudo bem, espero que levem minha sugestão de melhorar a visualização dessas informações na plataforma. Mas, no mais é isso, minhas dúvidas estão sanadas, obrigado.

Atendente: Por nada! Ajudo em algo mais?

Cliente: Não, somente. Até mais.

**Output Exemplo:**

Classificações: CUSTOS, EXPERIÊNCIA DO USUÁRIO

Input:
{conversa_text}
'''

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0.2,
        "top_p": 0.80,
    }

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    }

    def generate_classificacao():
        try:
            vertexai.init(project="nomedoproejtogoogle", location="localservidorgoogle")
            model = GenerativeModel(
                "gemini-1.5-flash-001",
                system_instruction=[
                    """Você é uma IA assistente especializada em classificar transcrições de chat entre um atendente e um cliente, com base em uma lista predefinida de classificações. Seu objetivo é precisão e acurácia quando for classificar uma conversa."""]
            )
            responses = model.generate_content(
                [prompt],
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=True,
            )
            # Coleta os valores gerados do gerador
            response_text = ""
            for response in responses:
                response_text += response.text
            return response_text
        except Exception as e:
            return json.dumps({"error": str(e)})

    resposta = generate_classificacao()
    # Encerrando contagem de duração do código
    end_time = time.time()
    # Atribuindo tempo de duração em segundos do código à variável 'duration_time'
    duration_time = end_time - start_time
    print("Tempo de duração:", duration_time)
    if isinstance(resposta, str):
        return resposta  # Se a resposta já for uma string JSON
    else:
        return json.dumps(json.loads(resposta.text))  # Converta a resposta para JSON