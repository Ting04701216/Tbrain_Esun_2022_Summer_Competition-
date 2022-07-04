import re

def phone_to_bopomo(d, string):
    """
    Args:
        d: A mapping dict from Phone to BoPoMo. e.g. {'p': 'ㄅ', ..., 'ttss3': 'ㄓˇ '}
        string: Phonetic string represented by Phone.
        
    Examples:
        string = 'k_h ax3 n ax N2 t aU3 ttss4 p u:2 ss4 p_h aU4 m O:4 ts aI4 s6 j A: n4'
        print(phone_to_bopomo(d, string))  # 'ㄎㄜˇ ㄋㄥˊ ㄉㄠˇ ㄓˋ ㄅㄨˊ ㄕˋ ㄆㄠˋ ㄇㄛˋ ㄗㄞˋ ㄒㄧㄢˋ'
    """
    
    str1 = re.sub("A: n", "A:+n", string)
    str1 = re.sub("ax n", "ax+n", str1)
    str1 = re.sub("A: N", "A:+N", str1)
    str1 = re.sub("ax N", "ax+N", str1)
    
    out_list = []
    for i in str1.split():
        try:
            str_bopomo = d[i]
        except:
            str_bopomo = i + ' '
        out_list.append(str_bopomo)
    out = ''.join(out_list).rstrip()
    
    return out
