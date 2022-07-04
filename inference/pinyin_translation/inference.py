from fairseq.models.transformer import TransformerModel
from src import translate_utils

model_path = 'models/all/checkpoints'
model_pinyin2zh = TransformerModel.from_pretrained(model_name_or_path=model_path, checkpoint_file='checkpoint_best.pt', data_name_or_path='../../../data/all/data-bin')


sentence_list = ['進而 形成 新 的 傳統',
  '進而 形成 心得 傳統',
  '金額 行程 新 的 傳統',
  '進而 形成 心 的 傳統',
  '金額 形成 新 的 傳統',
  '進而 形成 心地 傳統',
  '金額 行程 心得 傳統',
  '金額 行程 心 的 傳統',
  '金額 形成 心得 傳統',
  '金額 行程 心地 傳統']

phoneme_sequence_list = ['ts6 j ax n4 axr2 s6 j ax N2 ttss_h ax N2 s6 j ax n1 t ax5 ttss_h w A: n2 t_h w ax N3',
  'ts6 j ax n4 axr2 s6 j ax N2 ttss_h ax N2 s6 j ax n1 t ax2 ttss_h w A: n2 t_h w ax N3',
  'ts6 j ax n1 ax2 s6 j ax N2 ttss_h ax N2 s6 j ax n1 t ax5 ttss_h w A: n2 t_h w ax N3',
  'ts6 j ax n4 axr2 s6 j ax N2 ttss_h ax N2 s6 j ax n1 t ax5 ttss_h w A: n2 t_h w ax N3',
  'ts6 j ax n4 axr2 s6 j ax N2 ttss_h ax N2 s6 j ax n1 t ax5 ttss_h w A: n2 t_h w ax N3',
  'ts6 j ax n1 ax2 s6 j ax N2 ttss_h ax N2 s6 j ax n1 t ax2 ttss_h w A: n2 t_h w ax N3',
  'ts6 j ax n1 ax2 s6 j ax N2 ttss_h ax N2 s6 j ax n1 t ax5 ttss_h w A: n2 t_h w ax N3',
  'ts6 j ax n1 ax2 s6 j ax N2 ttss_h ax N2 s6 j ax n1 t ax5 ttss_h w A: n2 t_h w ax N3']

is_news = 1

phone2zh_translate = translate_utils.infer(sentence_list, phoneme_sequence_list, is_news, model_pinyin2zh)
print(phone2zh_translate)
