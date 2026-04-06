/////////////////////////////////////////////////////////////////MINHA ATIVIDADE /////////////////////////////////////////////////////////////
async function treinarEPrever() {

     // Pegando elementos da tela
         const textoStatus = document.getElementById("status");
        const textoResultado = document.getElementById("resultado");

    // Pegando valor digitado pelo usuario
         const minutosDigitados = Number(document.getElementById("minutos").value);

    // validação simples
        if (!Number.isFinite(minutosDigitados) || minutosDigitados <= 0) {
        textoStatus.innerText = "Status: Informe um numero valido de minutos (> 0).";
        textoResultado.innerText = "";
        return;
        }

         textoStatus.innerText = "Status: Treinando a IA...";

     // =========================
     // 1. CRIAR O MODELO (neuronio)
     // =========================
        const modelo = tf.sequential();
            modelo.add(tf.layers.dense({
            units: 1,
            inputShape: [1]
            }));

    // =========================
    // 2. COMPILAR O MODELO
    // =========================
        modelo.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
         });

    // =========================
    // 3. DADOS DE TREINO
    // X = minutos de exercício
    // Y = calorias queimadas (exemplo linear)
    // =========================
    const dadosEntrada = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    const dadosSaida = tf.tensor2d([50, 100, 150, 200], [4, 1]);

    // =========================
    // 4. TREINAMENTO
    // =========================
    await modelo.fit(dadosEntrada, dadosSaida, {
        epochs: 200
         });

        textoStatus.innerText = "Status: IA treinada!";

    // =========================
    // 5. PREVISÃO
    // =========================
    const previsao = modelo.predict(
     tf.tensor2d([minutosDigitados], [1, 1])
    );

    // Convertendo resultado para número
    const caloriasPrevistas = previsao.dataSync()[0];

    // Mostrando resultado na tela
    textoResultado.innerText =
         "Calorias Previstas: " + caloriasPrevistas.toFixed(2) + " kcal";
    }