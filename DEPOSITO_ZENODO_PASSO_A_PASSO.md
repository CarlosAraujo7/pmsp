# Depósito no Zenodo (DOI) e pré-registro público: passo a passo

Objetivo: transformar o diário com hashes (que hoje é só nosso) em um registro **com data verificável por terceiros**. O que prova a data é o DOI do Zenodo, que é gerado a partir de uma *release* do GitHub; o conteúdo da release inclui `diario_pesquisa.md` e todos os `results/entrada*_prereg_hashes.txt`.

## O que já está pronto no repositório
- `CITATION.cff` (título, autor, URL do repositório; o Zenodo lê esse arquivo para os metadados).
- Licença MIT (`LICENSE`), código, testes, diário, hashes de pré-registro, scripts de análise e auditoria.
- Artigo e suplemento (fontes `.tex`) em `paper/`.

## Passos (cerca de 10 minutos, só você pode fazer)
1. Entre em https://zenodo.org com sua conta do GitHub e ative o repositório `CarlosAraujo7/pmsp` em *Account → GitHub*.
2. **Antes** de criar a release, confira o `CITATION.cff` (nome, afiliação, e-mail, ORCID se tiver) e atualize `repository-code` se mudar o nome do repositório.
3. No GitHub: *Releases → Draft a new release*; tag `v1.0.0` (a mesma `version` do `CITATION.cff`); título "PMSP learnheuristics: code, data and analysis (pre-registration record)"; publique.
4. O Zenodo arquiva a release e gera dois DOIs: o da versão e o "conceito" (que sempre aponta para a última versão). Use o **DOI de conceito** no artigo.
5. Copie o DOI para: `paper/short/backmatter.tex` (Data and Code Availability, no lugar de "a DOI will be added on deposit"), `CITATION.cff` (campo `doi:`) e `README.md`; faça um novo commit.
6. Diga no artigo que o diário é o registro das hipóteses e que o depósito datado pelo Zenodo ancora as datas dos hashes. Frase sugerida: *"The research diary, with the hypotheses, decision rules, predictions and md5 hashes recorded before each analysis, is archived at Zenodo (DOI …); the deposit date anchors the record."*

## Cuidados
- Os dados de terceiros (benchmarks) **não** são redistribuídos: estão no `.gitignore` e o README indica as fontes.
- A release inclui o diário inteiro, com os erros e as previsões que falharam; é isso que dá valor ao registro.
- Depois de publicada, a release do Zenodo é imutável; correções entram numa nova versão (v1.0.1) e o DOI de conceito continua o mesmo.
- Se preferir um registro formal de pré-registro, o OSF Registries aceita o mesmo pacote (diário + hashes) e emite um registro com data; o Zenodo é suficiente para a finalidade deste trabalho.
