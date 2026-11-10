// WP-Spam weights, written by hand. `node gen/generate.mjs` turns this file into weights/spam.json.
//
// Every row of the embedding is a sparse vector over the named dims below. A row says what kind of evidence a word is
// ("this is a pharmacy word", "this is how colleagues talk"), never "this word means spam": the layers above decide how
// much each kind of evidence counts. Rows are general knowledge of how spam, scams, phishing and ordinary mail are
// written, in English and Portuguese; no row was read off the datasets (no dataset-specific names or addresses).
//
// The file has two parts. The first holds the core concepts, shapes, URL features, sentence roles and neurons; the
// second builds the network around them: the character layer, the letter-pair layer, the wider vocabulary, the
// attention heads, the extended roles and the MLP. LOG.md tells in which order they were written and why.

export const UNSPACE = true;   // "http : / / www . site . com" (pre-tokenised corpora) is rejoined into a link
export const FUZZY = true;     // doubled letters are ignored when matching ("vicoodiin" -> vicodin)

// ---------------------------------------------------------------- lexicon
// G(row, words): `stem*` fires at a word start, `~stem` anywhere inside a word, `a_b_c` is a phrase, a plain word
// fires on the whole word.
const LEX = {};
function G(row, words) {
  for (let w of words.trim().split(/\s+/)) {
    w = w.replace(/_/g, ' ');
    const r = (LEX[w] = LEX[w] || {});
    for (const [d, x] of Object.entries(row)) r[d] = +((r[d] || 0) + x).toFixed(3);
  }
}

// ================================================================ part 1: core concepts
const BASE_DIMS = [
  // spam-side evidence
  'money', 'prize', 'urgent', 'account', 'market', 'pharma', 'adult', 'invest', 'software', 'replica', 'advfee',
  'unsub', 'cta', 'html', 'smsspam', 'mobilemoney', 'salut', 'you',
  // shapes
  'excl', 'currency', 'percent', 'number', 'bignum', 'phone', 'shortcode', 'pence', 'url', 'email', 'allcaps',
  'obfuscated', 'random', 'decor',
  // ham-side evidence
  'work', 'energy', 'personal', 'tech', 'academic', 'list', 'quote',
  // written by the attention heads
  'amount', 'ctaLink', 'youWin', 'youAcct',
  // URL features
  'urlBad', 'urlGood',
  // more concepts
  'receipt', 'sendVerb', 'payReq', 'news', 'greet', 'func', 'funcRatio', 'accountMail',
  'solicited', 'capName', 'named', 'junk', 'junkRatio', 'urlBare', 'addressee', 'enhance', 'enhanceWeak', 'dating',
  // written by the sentence layer (r_ totals, s_ shares of sentences) and the message layer (a_ arcs)
  'r_offer', 'r_ask', 'r_contact', 'r_story', 'r_notice', 'r_chat', 'r_work', 'r_news',
  's_offer', 's_ask', 's_contact', 's_chat', 's_work', 's_news', 's_notice',
  'a_offerAsk', 'a_askContact', 'a_offerContact', 'a_storyOffer', 'a_storyAsk',
  // global
  'length', 'words', 'urlOnly', 'hasUrl',
];


// --- money: talk of sums, payments, wealth. Ham talks about money too (business mail), so its output weight is small.
G({ money: 1 }, `money cash dollar* usd euro* fund funds payment* paid pay payout wire bank* million* billion* thousand*
  income* earn* profit* rich wealth* fortune* sum loan cheque check_payment
  dinheiro pagamento* transferencia* transfer* valor saldo meticais mzn deposito* depositar levantar levantamento`);
// --- prize: something given for nothing, a win, a lottery.
G({ prize: 1 }, `won win wins winner winners winning prize* lottery lotto jackpot reward* bonus* gift* free freebie
  selected lucky award* awarded sweepstake* voucher* coupon* giveaway claim claimed draw
  guaranteed premio premios ganhou ganhar vencedor* parabens sorteio sorteado gratis oferta brinde`);
G({ prize: 1.5 }, `you_won you_have_won youve_won have_won been_selected you_are_a_winner claim_your claim_now
  free_entry cash_prize prize_draw`);
// --- urgent: pressure to act before thinking.
G({ urgent: 1 }, `urgent* urgently immediately immediate instant* instantly hurry expire* expir* limited
  deadline_today now_only today_only last_chance final_notice act_now asap_please suspend* suspension
  within_24 within_48 24hrs 48hrs 24_hours 48_hours imediatamente urgente rapido ultima ultimo`);
// --- account: the language of credential and payment phishing.
G({ account: 1 }, `account accounts verify verified verification confirm confirmation password* passcode login logon
  signin sign_in log_in credential* locked unlock* unusual suspicious_activity unauthorized identity ssn pin otp
  billing invoice* refund* mailbox webmail quota storage_full administrator helpdesk help_desk security_alert
  conta senha verificar verifique confirme bloqueada bloqueado codigo pin_code`);
G({ account: 1.5 }, `your_account your_password verify_your confirm_your update_your account_has account_will
  your_mailbox your_card your_bank sua_conta seu_pin`);
G({ account: 0.7 }, `paypal ebay amazon netflix apple icloud itunes microsoft outlook office365 hotmail dhl fedex usps
  irs hmrc chase wellsfargo barclays lloyds natwest hsbc santander`);
// --- market: a sales pitch.
G({ market: 1 }, `offer* discount* save saving* sale sales buy buying order ordering shop shopping store product*
  quality guarantee* satisf* best lowest cheap* cheapest special exclusive promo promos promotional amazing incredible
  unbelievable miracle opportunit* bargain clearance wholesale retail brand deal_of shipping delivered_to
  desconto promocao loja compre`);
G({ market: 1.5 }, `buy_now order_now order_today shop_now limited_time special_offer best_price lowest_price
  risk_free no_obligation money_back satisfaction_guaranteed free_shipping free_trial great_deal you_save save_up
  up_to work_from_home make_money extra_income earn_money as_seen 100_free`);
// --- pharma: the pharmacy spam vocabulary.
G({ pharma: 1 }, `viagra cialis levitra pill pills pharmacy* pharmacies meds medication* prescription* rx vicodin xanax
  valium tramadol phentermine ambien soma propecia meridia diet* slim weight_loss lose_weight fat enlarge*
  enhancement penis erection* erectile potency herbal generic generics impotence libido orgasm* hgh
  antibiotics painkiller* no_prescription online_pharmacy`);
// --- adult: adult and dating spam.
G({ adult: 1 }, `sex sexy sexual porn* xxx nude* naked horny hottest girls babes dating singles lonely adult erotic
  hookup milf webcam cams slut* hardcore teen* lesbian* fetish`);
// --- invest: stock touting, loans and mortgages.
G({ invest: 1 }, `stock stocks shares invest* investor* mortgage* refinanc* loan loans lender* credit debt* apr approved
  approval homeowner* equity forex otc ticker symbol soar* explode* skyrocket* breakout undervalued
  pre_approved preapproved rates insurance quote`);
G({ invest: 1.5 }, `strong_buy hot_stock stock_alert low_rate bad_credit no_credit credit_check current_price
  stock_symbol investor_alert featured_profile penny_stock penny_stocks hidden_gems price_target`);
// --- the same pitches in Spanish and German (only the pitch words; ordinary words stay neutral)
G({ market: 1 }, `gratis oferta ofertas descuento compra kostenlos angebot rabatt jetzt`);
G({ cta: 1 }, `haga_click click_aqui pulse klicken sie_hier`);
G({ adult: 1 }, `lust geil sexo`);
// --- software: cheap OEM software spam.
G({ software: 1 }, `oem software adobe photoshop acrobat illustrator macromedia autocad windows xp vista office
  coreldraw norton antivirus licen* serial crack* cracked cd dvd`);
// --- replica: knock-off goods.
G({ replica: 1 }, `replica replicas rolex watches luxury designer handbag* swiss cartier gucci vuitton omega breitling
  prada chanel tiffany`);
// --- advfee: the advance-fee story (inheritance, a transfer, a stranger asking for a partner).
G({ advfee: 1 }, `beneficiar* deceased inherit* kin barrister confidential* strictly consignment diplomat* trunk
  dormant abandoned unclaimed widow orphan refugee nigeria* lagos ghana accra abidjan ivory cote dakar sierra leone
  burkina togo benin liberia zimbabwe senegal modalities trustworthy compliment* claimant funds_transfer
  next_of_kin business_proposal foreign_account our_client late_husband late_father`);
G({ advfee: 0.4 }, `attorney lawyer foreign partner* estate minister ministry government petroleum crude contract*
  mutual trust* honest* reliable assist assistance cooperation percent sharing proposal transaction* bank_account
  god bless blessing* christian* muslim sincerely_yours`);
G({ advfee: 1.5 }, `next_of_kin late_client huge_sum security_company dormant_account 30_percent 40_percent`);
// --- salut: openings that address no one in particular.
G({ salut: 1 }, `dear_friend dear_sir dear_madam dear_customer dear_user dear_member dear_winner dear_beneficiary
  valued_customer dear_account attention_beneficiary greetings_to caro_cliente prezado_cliente
  estimado_cliente`);
// --- unsub: mass-mail footers.
G({ unsub: 1 }, `unsubscrib* removal optout opt_out be_removed remove_me no_longer_wish to_unsubscribe this_mailing
  receive_this future_mailings mailing_list click_to`);
// --- cta: asking the reader to do something now.
G({ cta: 1 }, `click call visit reply text txt send claim apply register signup subscribe download contact
  click_here click_below call_now visit_our go_to follow_the reply_with text_back send_your
  clique ligue envie responda acesse aceda carregue devolva devolver manda mande`);
// --- html: markup words that only appear when the mail is HTML.
G({ html: 1 }, `nbsp href src img bgcolor font td tr tbody table div span br colspan cellpadding cellspacing valign
  align width height border face color style html body meta`);
// --- smsspam: premium-rate SMS vocabulary.
G({ smsspam: 1 }, `ringtone* ringtones tones msg msgs mob nokia camcorder polyphonic chatline
  landline t_c t_cs tc tcs pobox po_box 150p cust customer_care
  per_msg per_min std_rates network_operator unsub 12hrs`);
G({ smsspam: 0.5 }, `mobile camera chat stop subscription customer_service valid tone`);
G({ smsspam: 1.5 }, `txt_stop text_stop reply_stop send_stop to_opt claim_code your_mobile mobile_number`);
// --- mobilemoney: Mozambican mobile-money scam vocabulary (wrong-transfer and fake-agent scams).
G({ mobilemoney: 1 }, `engano enganei devolva devolver devolve`);
G({ mobilemoney: 1.5 }, `por_engano enviei_por mandei_por meu_dinheiro devolver_o este_numero para_este vem_em em_nome
  neste_numero neste_nr nesta_conta neste_numero esse_nr este_nr`);
// --- you: addressing the reader. Both sides do it; spam does it more.
G({ you: 1 }, `you your yours yourself u ur voce seu sua teu tua vc`);

// --- work: how colleagues write to each other.
G({ work: 1 }, `meeting* meet attached attach* schedul* project* report* review* team discuss* agenda draft* memo
  forecast* budget* presentation* regards thanks thx fyi cc question* issue* plan* status minutes
  conference_call let_me let_us lets please_let original_message forwarded_by
  colleague* manager* department office* tomorrow monday tuesday wednesday thursday friday spreadsheet* excel
  analysis analys* model* data version file files comments feedback approve* process`);
G({ work: 1.5 }, `let_me_know talk_to please_review see_attached as_discussed per_our our_meeting your_comments
  the_attached i_will we_will i_think we_need i_have`);
// --- energy: the industry vocabulary of trading and operations mail.
G({ energy: 1 }, `gas power energy pipeline capacity volume* trading trader* desk counterpart* nomination* deal deals mmbtu
  mw mwh curve* risk position* settlement* storage transport* utility utilities electricity plant* generation
  operations logistics contract_terms`);
// --- personal: chat between people who know each other.
G({ personal: 1 }, `hey lol haha hehe ok okay k yeah yea ya yup hmm gonna wanna gotta dunno sorry tonight
  dinner lunch sleep mum mom dad bro sis lor
  abt nt dat dis neva nite gud luv coz cos thk tmr tmrw wat wot wil
  estou vou onde quando casa amanha hoje ontem jantar mae pai amor beijo* bom boa dia noite ta tudo bem obrigad*
  sim nao ja mano saudades chegar chegou aqui ai kkk kkkk`);
// --- tech: software and mailing-list discussion.
G({ tech: 1 }, `linux unix kernel patch* bug bugs fix fixed compile* compil* debug* error* warning* function* variable*
  module* library librar* install* config* server* cvs svn git commit* branch* release* build* source binary
  package* rpm apt deb debian ubuntu redhat fedora suse gentoo freebsd perl python java javascript ruby gcc api
  documentation docs faq howto manual wiki repository devel developer* maintainer* upstream regression segfault
  stack trace log logs output input script* shell bash terminal directory folder path default option* argument*
  parameter* integer array object* method* thread* memory cpu disk driver* hardware emacs vim xml sql mysql
  apache samba spamassassin procmail mutt sourceforge`);
// --- academic: university and research mail.
G({ academic: 1 }, `linguistic* language* syntax semantic* phonolog* morpholog* grammar* corpus corpora dialect* verb* noun*
  sentence* lexical discourse pragmatic* university universities department professor prof workshop symposium
  abstract* paper* submission* proceedings journal research* lecture* seminar* colloquium student* phd
  dissertation thesis summary query references fax edu`);
// --- list: mailing-list machinery (a list post is someone's message to their community).
G({ list: 1 }, `listinfo mailman archives digest wrote writes posted list_info
  on_behalf in_reply reply_to`);
G({ list: 1.5 }, `original_message forwarded_message`);

// ---- receipts, news, salad, URLs, e-mail-account phishing
G({ prize: 0.4 }, `congratulation* congrats`);
// --- pharma by euphemism: the same pitch without the drug names.
G({ pharma: 1 }, `stud manly manhood masculine pecker lovemaking stamina potent inches lengthening largeness enlargement
  bigger_size last_longer in_bed her_crazy drive_her girlfriend_will satisfy_her side_effects your_package
  wonder_pill sexual_performance love_life bedroom ejaculation`);
G({ replica: 1 }, `imitation imitations watch_lovers knockoff knock-off`);
G({ adult: 1 }, `pretty_girl pictures_of my_photos lonely_girl hot_girls`);
// --- accountMail: the e-mail-account phishing story (your mailbox is full, messages on hold, files waiting).
G({ accountMail: 1 }, `mailbox quota exceeded exceed exceeds deactivat* reactivat* re-activate pending on_hold
  incoming_messages storage_capacity almost_full cpanel kindly upgrade_your validate_your shared_a has_shared
  sent_you view_document review_document wetransfer docusign e-fax efax voicemail outstanding
  will_be_deleted be_suspended be_closed be_terminated within_hours`);
// --- greet: opening words; "Dear jose@example.org" addresses the reader by an address, not a name (head dearEmail).
G({ greet: 1 }, `dear hello hi attention`);
// --- sendVerb: asking for money to be sent (head payReq links it to a phone or account number).
G({ sendVerb: 1 }, `manda mande mandar envia envie enviar enviame transfere transfira transferir deposita deposite
  send transfer wire deposit`);
// --- receipt: a completed transaction being reported (past tense, a reference, a new balance). Banks and
// mobile-money services send these by the million; a scam asks, a receipt tells.
G({ receipt: 1 }, `confirmado recebeste transferiste levantaste compraste pagaste efectuada efectuado sucesso
  novo_saldo o_teu teu_saldo taxa imposto referencia operacao montante informamos informa efectuou deduzido
  debitado creditado liquidar transaccoes levantamento com_sucesso foi_efectuada se_desconhece efetuada efetuado
  autorizar autorizacao introduza compra_movel you_have_received have_received
  transaction_id has_been_credited new_balance your_balance reference_number receipt`);
// --- news: reported speech and the vocabulary of journalism. Newsletters and news digests are ham.
G({ news: 1 }, `said says told according reported reporter* reuters ap afp bbc press spokesman spokeswoman officials
  analysts investigators court judge police election senate congress parliament republican* democrat* labour
  tory minister ministers president prime war troops killed died report_says headline* edition newsletter
  correspondent journalist* editorial columnist interview* interviewed article* story stories blog* posted
  columns opinion`);
// --- more ham: research meetings and their calls for papers.
G({ academic: 1 }, `conference conferences registration program programme call_for for_papers cfp submission
  deadline deadlines accepted keynote tutorial* track committee chair* organizers`);
G({ tech: 1 }, `svn revision modified commit trunk regexp postfix sendmail dns smtp imap header headers
  bugzilla changelog diff`);

// ---- euphemism, dating, solicited mail, names in greetings
// --- enhance: the male-enhancement pitch, which mostly avoids drug names and speaks in euphemism.
G({ enhance: 1 }, `stiff hardness rod manhood masculine manly stud stamina erection* ejaculat* potency
  libido orgasm* lengthening largeness enlarg* intimacy pecker penis member_size male_power male_enhancement
  more_length greater_length extra_inches more_inches her_pleasure bigger_and longer_and harder_and stay_hard
  love_muscle love_life love_making lovemaking in_bed at_bed all_night satisfy_her her_crazy your_lady your_beloved
  your_partner your_gf her_happy make_her drive_her last_longer size_matters your_size new_size more_length
  your_tool your_rod your_warrior little_warrior secret_weapon real_man true_man male_merits male_power
  new_dimension the_dimension this_remedy new_formula advanced_formula pills_for performance_anxiety stick_of`);
G({ pharma: 1 }, `oxycontin oxycodone hydrocodone percocet codeine adderall ritalin klonopin ativan lorazepam alprazolam
  diazepam zolpidem sildenafil tadalafil vardenafil finasteride psychotropic* analgesic* antibiotic* pharmaceutical*
  apotheke apoteke medz medicat* drugstore anti-ed antied eds`);
// --- enhanceWeak: ordinary words the same pitch leans on. They count only inside the enhancePitch neuron, which news,
// tech and work talk switch off, so "a longer battery life" in a newsletter does not fire it.
G({ enhanceWeak: 1 }, `pleasure pleasures bigger longer harder stronger thicker massive inches length size girth
  remedy formula weapon warrior giant proud admired confidence lady beloved gf girlfriend satisfy satisfied lover
  dimension dimensions muscle power powerful hard`);
// --- dating: "a nice girl would like to chat", "looking for a man", "see my photos".
G({ dating: 1 }, `nice_girl lonely_girl russian_girl looking_for_man looking_for_a meet_your your_lover meet_singles
  chat_with my_pictures my_photos pictures_of photos_of see_my my_profile become_a_member single_women
  hot_singles local_singles date_tonight new_friends i_am_tired write_me email_me_at my_friend's`);
// --- solicited: the machinery of mail people signed up for (a store's newsletter, a society's bulletin).
// Spam fakes some of it, so it counts less than conversation does.
// (Legal footers - "privacy policy", "all rights reserved", "manage your subscription" - are copied into spam
// wholesale, so they are not evidence either way.)
G({ solicited: 1 }, `address_book view_online view_this trouble_viewing having_trouble
  you_subscribed you_signed signed_up you_are_receiving are_receiving receiving_this
  in_this_issue this_issue issue_no member_services
  newsletter newsletters edition bulletin weekly monthly daily_update digest url headlines
  todays_headlines tip_of the_day`);
// --- addressee: words that address nobody in particular ("Dear Friend") - neutral, but they are not a name.
G({ addressee: 1 }, `friend friends sir madam customer customers user users member winner beneficiary client colleague
  colleagues partner valued sirs gentlemen all everyone team reader readers subscriber subscribers`);
// --- list: replies across languages, and mailing-list footers.
G({ list: 1 }, `schrieb escreveu escribio ecrit wrote_on said_on google_groups groups_google post_to
  this_list the_list list_archive mailing-list`);
G({ personal: 1 }, `pls plz ppl`);
G({ greet: 1 }, `hey dear_mr dear_ms dear_dr`);

// ---------------------------------------------------------------- function words (for the funcRatio feature)
export const FUNCTION_WORDS = `the of and to a in is it that for on was with as at by be this are or from but not have
  had has were which an they he she we i his her their there been would will can could if so what all about when who
  o os as de do da dos das que e em um uma para com por se na no nas nos mas mais como foi ao ele ela eu ja muito
  el la los las y en por con es lo del al su para un una der die und den das ist nicht ich sie es mit von zu
  my why try by mr mrs ms dr st pm tv pc cd dvd www http com html lt gt amp nbsp quot`.split(/\s+/).join(' ');

const BASE_RATIOS = [
  // share of words that are function words; prose in any of these languages sits near 0.4, word salad and
  // letter-fragment obfuscation near 0.1. Short texts (< 15 words) get the neutral value 0.4.
  { out: 'funcRatio', num: 'func', minWords: 15, empty: 0.4 },
  // share of words that are 2-4 letter consonant clusters ("Pe lbd nis En ir large ljb ment"): junk inserted to
  // break a word into pieces a filter cannot read. Ordinary text has almost none.
  { out: 'junkRatio', num: 'junk', minWords: 15, empty: 0 },
];

// ---------------------------------------------------------------- shapes (one row per token shape)
const BASE_SHAPES = {
  url: { url: 1 },
  email: { email: 1 },
  number: { number: 1 },
  bignum: { number: 1, bignum: 1 },        // >= 10,000: a sum, a prize, a population
  phone: { number: 1, phone: 1 },          // 9+ digits with a phone-like start
  shortcode: { number: 1, shortcode: 1 },  // a 4-6 digit number right after "text/send/call ... to"
  pence: { pence: 1, currency: 1 },        // 150p, 50ppm
  'p!': { excl: 1 },
  'p$': { currency: 1 },
  'p£': { currency: 1 },
  'p€': { currency: 1 },
  'p%': { percent: 1 },
  'p*': { decor: 1 },
  'p>': {},
  quoteMark: { quote: 1 },                  // a line starting with ">": quoting someone's earlier mail
  inQuote: { quote: 0.2 },                  // words inside quoted lines
  allcaps: { allcaps: 1 },
  obfuscated: { obfuscated: 1 },            // v1agra, c-i-a-l-i-s: hiding a word from filters
  random: { random: 1 },                    // hash-buster strings
  reply: { quote: 0.5 },                    // "Re:" opening the subject
  func: { func: 1 },                        // a function word (counted for funcRatio)
  junk: { junk: 1 },                        // a consonant-only fragment (counted for junkRatio)
  capName: { capName: 1 },                  // a capitalised word the lexicon does not know: usually a name
  listTag: { list: 1.5 },                   // "[listname]" opening the subject
  'p~': { list: 0.1 },                      // "--~--~--" group footers
  emoticon: { personal: 1.5 },              // ":)", ";-)", ":D": people chatting, rarely mass mail
};

// ---------------------------------------------------------------- URL features (see engine urlFeatures)
const BASE_URL = {
  partScale: 0.5,   // words inside a URL count half (".../account/verify/...")
  rows: {
    ip: { urlBad: 1.5 }, at: { urlBad: 1.5 }, dashes: { urlBad: 0.8 }, subdomains: { urlBad: 0.8 },
    long: { urlBad: 0.4 }, verylong: { urlBad: 0.4 }, badtld: { urlBad: 1.2 }, freehost: { urlBad: 1.0 },
    shortener: { urlBad: 1.0 }, hostdigits: { urlBad: 0.6 }, odd: { urlBad: 0.8 }, keyword: { urlBad: 0.7 },
    brandelsewhere: { urlBad: 1.5 }, http: { urlBad: 0.1 }, script: { urlBad: 0.1 },
    article: { urlGood: 1 }, bare: { urlBare: 1 },   // a bare domain is ordinary as a lone link, not inside a message
    cmsPath: { urlBad: 1.5 },        // a page planted inside a hacked site's CMS folders
    randomPath: { urlBad: 1.0 },     // machine-made path segments
    deep: { urlBad: 0.4 },
    document: { urlGood: 1.0 },      // a link to a PDF or document
    institution: { urlGood: 1.0 },   // .edu / .gov / .ac.xx
    repeatSeg: { urlBad: 1.0 },      // "/test/test/", "/x7Q/x7Q/": paths made by a phishing kit, not by a site's author
  },
  sets: {
    badTld: 'tk ml ga cf gq xyz top club online site info biz pw cc buzz icu work support click link'.split(' '),
    freeHost: '000webhostapp.com wixsite.com firebaseapp.com web.app herokuapp.com sites.google.com glitch.me netlify.app vercel.app ngrok.io repl.co azurewebsites.net appspot.com godaddysites.com square.site yolasite.com webflow.io duckdns.org no-ip.org myftp.org'.split(' '),
    shortener: 'bit.ly tinyurl.com goo.gl t.co ow.ly is.gd cutt.ly rebrand.ly shorturl.at tiny.cc'.split(' '),
    keyword: 'login signin logon verif account updat secur confirm banking webscr ebayisapi password wallet recover unlock billing invoice validat'.split(' '),
    brand: 'paypal apple icloud itunes microsoft office365 outlook hotmail onedrive sharepoint amazon netflix ebay facebook instagram whatsapp gmail dropbox docusign chase wellsfargo bankofamerica hsbc santander barclays lloyds natwest dhl fedex usps coinbase binance metamask blockchain linkedin yahoo aol adobe webmail'.split(' '),
  },
};

// ---------------------------------------------------------------- hidden layer (ReLU over pooled log-counts)
const BASE_HIDDEN = [
  { name: 'urlVerdict', w: { urlBad: 1, urlGood: -1, urlBare: -0.8, urlOnly: 3 }, b: -3,
    why: 'a lone link is judged by its own shape; inside a message a link must be very bad to count this much' },
  { name: 'advanceFee', w: { advfee: 0.6, amount: 1, money: 0.3, salut: 1, urgent: 0.5, news: -1, work: -0.5, energy: -0.5 }, b: -2.5,
    why: 'a stranger, a large sum and a confidential deal together are the advance-fee story' },
  { name: 'phishing', w: { account: 1, youAcct: 1, urgent: 0.7, ctaLink: 0.7, urlBad: 0.7, salut: 0.5, accountMail: 1 }, b: -2,
    why: 'your account + pressure + a link is credential phishing' },
  { name: 'prizeCall', w: { youWin: 1, prize: 0.5, cta: 0.5, phone: 0.7, shortcode: 1, pence: 0.7 }, b: -1.5,
    why: 'you have won + call/text a number is the premium-rate prize scam' },
  { name: 'pitch', w: { market: 0.8, unsub: 0.8, html: 0.5, percent: 0.5, ctaLink: 0.5, news: -0.7, solicited: -0.7, named: -1 }, b: -1.5,
    why: 'a sales pitch with a footer and markup is bulk marketing' },
  { name: 'junkText', w: { junkRatio: 25 }, b: -3,
    why: 'more than ~12% consonant fragments: words chopped up with junk between the pieces (texting drops vowels too, but less)' },
  { name: 'enhancePitch', w: { enhance: 1, enhanceWeak: 0.6, hasUrl: 0.7, excl: 0.3, you: 0.3, news: -1, tech: -1, work: -0.5, solicited: -0.5, list: -0.5 }, b: -1.6,
    why: 'the euphemistic enhancement pitch: several of its words, usually with a link' },
  { name: 'salad', w: { funcRatio: -10 }, b: 1.7,
    why: 'text with too few function words is word salad or chopped-up words, not something a person wrote' },
  { name: 'shortLink', w: { hasUrl: 1.6, words: -0.45, urlOnly: -3, solicited: -1.5, news: -0.7 }, b: 0.6,
    why: 'a short message whose point is a link (few words, then a URL) is how bulk spam is written' },
  { name: 'payRequest', w: { payReq: 1.5, mobilemoney: 1, receipt: -1.5 }, b: -0.5,
    why: 'asked to send money to a number, and nothing reports a completed transaction' },
  { name: 'conversation', w: { quote: 1.2, personal: 0.5, work: 0.5, list: 0.5 }, b: -1.5,
    why: 'quoting, chatting and working together is a conversation between people who know each other' },
];

// ---------------------------------------------------------------- sentence layer (core roles)
// Each sentence (split at . ! ? and line breaks) is pooled on its own: log1p of the sum of its tokens' vectors. A role
// unit is a clamped ReLU, min(1, max(0, b + w.P)): one clear word of the kind gives about 0.6, two give 1.
const BASE_ROLES = [
  { name: 'offer', b: -0.3, why: 'something is on offer: a prize, a sum, a product, a drug, a date',
    w: { prize: 1.2, money: 0.8, amount: 1.2, market: 1.0, pharma: 1.4, enhance: 1.4, enhanceWeak: 0.4, invest: 1.0,
      software: 1.0, replica: 1.4, adult: 1.0, dating: 1.2, bignum: 0.5, currency: 0.6, percent: 0.5, youWin: 1.0 } },
  { name: 'ask', b: -0.3, why: 'the reader is told to do something: click, call, reply, send, verify, hurry',
    w: { cta: 1.2, sendVerb: 1.2, urgent: 0.8, account: 0.8, accountMail: 1.0, payReq: 1.5, smsspam: 0.8,
      mobilemoney: 1.0, youAcct: 1.0, excl: 0.3 } },
  { name: 'contact', b: -0.2, why: 'a way to act: a link, an address, a phone number, a shortcode',
    w: { url: 1.5, email: 1.5, phone: 1.5, shortcode: 1.5, ctaLink: 1.5 } },
  { name: 'story', b: -0.3, why: 'the advance-fee story: a stranger, a deceased client, a confidential transfer',
    w: { advfee: 1.2, salut: 1.0 } },
  { name: 'notice', b: -0.3, why: 'a transaction receipt or a subscribed bulletin',
    w: { receipt: 1.2, solicited: 1.0 } },
  { name: 'chat', b: -0.3, why: 'people who know each other: chat forms, quoting, a name in the greeting, a list',
    w: { personal: 1.2, quote: 1.2, named: 1.0, list: 0.8 } },
  { name: 'work', b: -0.4, why: 'work, trade, software, research', w: { work: 1.0, energy: 1.0, tech: 1.0, academic: 1.0 } },
  { name: 'news', b: -0.3, why: 'reported news', w: { news: 1.2 } },
];
// ---------------------------------------------------------------- message layer (attention across sentences)
// a_X: for each sentence with role `from`, the strongest `to` role within `window` sentences after it (same: the same
// sentence counts too), summed, log1p. A scam is an arc: offer -> ask -> contact; story -> offer -> ask.
const BASE_ARCS = [
  { name: 'offerAsk', from: 'offer', to: 'ask', window: 2, same: true },
  { name: 'askContact', from: 'ask', to: 'contact', window: 1, same: true },
  { name: 'offerContact', from: 'offer', to: 'contact', window: 3, same: true },
  { name: 'storyOffer', from: 'story', to: 'offer', window: 4, same: true },
  { name: 'storyAsk', from: 'story', to: 'ask', window: 5, same: false },
];
// ---------------------------------------------------------------- gate units (core)
const BASE_HIDDEN2 = [
  { name: 'channel', clamp: true, b: 0,
    w: { url: 2, email: 2, phone: 2, shortcode: 2, cta: 1, sendVerb: 1, account: 0.7, accountMail: 0.7, smsspam: 1,
      payReq: 1, ctaLink: 1, r_contact: 2, r_ask: 1 },
    why: 'is there any way for the reader to act on this message? (1 = yes)' },
  { name: 'noChannel', clamp: true, b: 1, h: { channel: -1 },
    why: 'nothing to click, call, reply to or send (corpora that strip links make this weaker evidence than it should be)' },
  { name: 'looseNoChannel', b: 0, w: { prize: 1, urgent: 1, youWin: 1, money: 0.6, excl: 0.4, you: 0.3 }, h: { channel: -3 },
    why: 'loose words (win, instant, money, you) with no way to act: cancel them. ReLU(loose - 3*channel) is the loose '
      + 'evidence when there is no channel and nothing when there is one; strong evidence (drugs, adult, salad) is not gated' },
  { name: 'scamArc', b: -0.5, w: { a_offerAsk: 1, a_askContact: 1, a_offerContact: 1, a_storyOffer: 1, a_storyAsk: 1 },
    why: 'the roles appear in the order a pitch is built' },
  { name: 'spamShape', b: -0.5, w: { s_offer: 1, s_ask: 1, s_contact: 0.5, s_chat: -1, s_work: -1, s_news: -1, s_notice: -1 },
    why: 'most sentences of the message offer or ask' },
  { name: 'hamShape', b: -0.4, w: { s_chat: 1, s_work: 1, s_news: 1, s_notice: 1, s_offer: -1, s_ask: -1 },
    why: 'most sentences of the message chat, work, report or notify' },
];

// ---------------------------------------------------------------- output layer
const BASE_OUT = {
  b: -1.2,
  w: {
    money: 0.5, prize: 1.2, urgent: 0.6, account: 0.6, market: 0.9, pharma: 2.2, adult: 1.4, invest: 0.8, software: 1.0,
    replica: 2.0, advfee: 0.8, unsub: 0.8, cta: 0.5, html: 0.8, smsspam: 1.5, mobilemoney: 0.8, salut: 0.8, you: 0.2,
    excl: 0.4, currency: 0.4, percent: 0.4, bignum: 0.4, phone: 0.8, shortcode: 1.5, pence: 1.5, url: 0.2,
    allcaps: 0.3, obfuscated: 1.5, random: 0.6, decor: 0.2,
    amount: 1.0, ctaLink: 0.8, youWin: 1.0, youAcct: 0.8, urlBad: 1.5, urlGood: -1.2,
    work: -1.0, energy: -0.8, personal: -0.9, tech: -1.2, academic: -1.2, list: -1.2, quote: -1.5,
    receipt: -2.5, news: -1.0, accountMail: 1.2, payReq: 0.8,
    solicited: -0.8, named: -1.2, enhance: 0.8, dating: 1.2,
  },
  hidden: { advanceFee: 1.5, phishing: 1.5, prizeCall: 1.2, pitch: 1.0, conversation: -1.0, salad: 1.2, shortLink: 1.5, urlVerdict: 2.0, junkText: 1.5, enhancePitch: 1.2,
    looseNoChannel: -0.6, scamArc: 0.4, spamShape: 1.5, hamShape: -1.5,
    payRequest: 1.5 },
};


// ================================================================ part 2: the network around them

// ---------------------------------------------------------------- dims
export const DIMS = [
  ...BASE_DIMS.filter(d => !['length', 'words', 'urlOnly', 'hasUrl'].includes(d)),
  // more spam-side concepts
  'crypto', 'gamble', 'extort', 'jobScam', 'parcel', 'seo', 'renewal', 'brand', 'health', 'debt',
  // relation words (read by heads; most have no output weight of their own)
  'deixis', 'obligation', 'timeUnit', 'price', 'question', 'clock', 'alphaRatio', 'unpron', 'unpronRatio', 'urlKnown',
  // more ham-side concepts
  'everyday', 'emotion', 'firstPerson', 'past', 'plan', 'timeRef', 'family', 'shared', 'topic',
  // more head outputs
  'deadline', 'priceTag', 'discount', 'impersonate', 'pressure', 'narrate', 'arrange', 'cryptoPay', 'extortPay',
  'jobPay', 'parcelFee', 'offerFree', 'chatQ', 'techScam', 'offerLink', 'threatLink', 'storyMoney', 'meetup',
  'familyTalk',
  // more sentence roles
  'r_threat', 'r_life', 's_threat', 's_life', 'a_threatAsk', 'a_threatContact',
  // global
  'length', 'words', 'urlOnly', 'hasUrl',
];

// ---------------------------------------------------------------- character embedding
// Each character is a vector over letters. A letter is itself (1). Digits and symbols used to disguise words count as
// the letter they imitate (0.9), and the Cyrillic letters that look like Latin ones count as those letters (0.9), so a
// filter written for "viagra" also reads "v1agra" and "vіagra". An "l" and an "i" are read as each other at 0.5
// ("pieased", "profiie"): 0.5 is nothing in a short word ("mail" is not "mall"), enough in a long one.
const CH = {};
for (const c of 'abcdefghijklmnopqrstuvwxyz0123456789\'') CH[c] = { [c]: 1 };
const GLYPH = { '0': 'o', '1': 'il', '3': 'e', '4': 'a', '5': 's', '7': 't', '@': 'a', '$': 's', '|': 'li' };
for (const [c, ls] of Object.entries(GLYPH)) { CH[c] = CH[c] || {}; for (const l of ls) CH[c][l] = 0.9; }
CH['8'].b = 0.6; CH['9'].g = 0.7;
CH.l.i = 0.5; CH.i.l = 0.5;
const CYR = { 'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'х': 'x', 'у': 'y', 'і': 'i', 'ј': 'j', 'ѕ': 's', 'к': 'k', 'м': 'm', 'т': 't', 'в': 'b', 'н': 'h' };
for (const [c, l] of Object.entries(CYR)) CH[c] = { [l]: 0.9 };
export const CHARS = CH;
export const TOL = { short: 0.5, long: 1.5 };   // filter tolerance: a = clamp((score - length + t) / t)

// ---------------------------------------------------------------- more lexicon
// G(row, words): `stem*` fires at a word start, `~stem` anywhere inside a word, `a_b_c` is a phrase, a plain word
// fires on the whole word.

// ---- brands that phishing imitates: their own dim, read by the impersonate head (a brand next to account talk).
G({ brand: 1 }, `paypal ebay amazon netflix apple icloud itunes microsoft outlook office365 hotmail dhl fedex usps irs
  hmrc chase wellsfargo barclays lloyds natwest hsbc santander bankofamerica citibank citi visa mastercard amex
  americanexpress docusign dropbox onedrive sharepoint adobe linkedin facebook instagram whatsapp gmail google yahoo
  aol coinbase binance steam spotify ups royalmail royal_mail canadapost auspost correios vodacom millennium wells_fargo
  bank_of_america american_express apple_id`);
G({ brand: 1 }, `bci standardbank standard_bank absa mpesa m-pesa emola mkesh nedbank fnb capitec itau bradesco nubank
  caixa cgd millenniumbcp novobanco revolut venmo cashapp zelle westernunion western_union moneygram aramex
  ctt correos laposte postnl dpd evri hermes gls`);

// ---- crypto: coins, wallets, and the doubling / recovery / investment scams built on them.
G({ crypto: 1 }, `~bitcoin btc ethereum eth cryptocurrenc* blockchain cryptocurrenc* altcoin* usdt tether
  dogecoin litecoin ripple xrp solana defi staking airdrop* nft nfts metamask trustwallet satoshi hashrate presale ico
  crypto cryptos
  wallet_address btc_address bitcoin_address seed_phrase private_key recovery_phrase secret_phrase double_your
  your_wallet crypto_wallet mining_pool cloud_mining trading_bot forex_signals signal_group`);
G({ crypto: 0.4 }, `wallet* token tokens mining miner* exchange coin coins ledger trezor`);

// ---- gamble: casinos and betting.
G({ gamble: 1 }, `casino* ~casino poker roulette blackjack baccarat slots jackpot* betting sportsbook bookmaker* wager*
  gambling gamble keno free_spins spins deposit_bonus welcome_bonus no_deposit bonus_code place_your_bet bet_now
  online_casino play_now`);
G({ gamble: 0.5 }, `bet bets odds bingo slot`);

// ---- extort: sextortion and blackmail ("I recorded you", "pay in bitcoin or I send it to your contacts").
G({ extort: 1 }, `recorded_you recorded_video your_webcam
  your_camera your_device your_contacts all_your_contacts your_friends_and contact_list adult_websites adult_sites
  porn_sites porn_site masturbat* compromising embarrassing humiliat* explicit_video video_of_you delete_the_video
  delete_everything i_will_delete erase_everything once_i_receive after_payment i_know_your your_password_is
  dont_try don't_try no_point police_won't do_not_reply_to report_this blackmail* extort*`);
G({ extort: 0.4 }, `hacked spyware trojan keylogger malware hacker hackers access_to_your your_system i_have_access full_access`);

// ---- jobScam: work-from-home and recruitment pitches that promise pay for little work.
G({ jobScam: 1 }, `work_from_home work_at_home home-based home_based data_entry mystery_shopper secret_shopper
  envelope_stuffing be_your_own your_own_boss own_boss financial_freedom passive_income residual_income
  mlm downline online_job online_jobs easy_money extra_cash make_money_online earn_up earn_from per_day_from
  no_experience no_experience_needed flexible_hours daily_pay weekly_pay paid_daily paid_weekly job_offer
  job_opportunity hiring_now we_are_hiring now_hiring apply_today apply_now part-time part_time parttime
  payment_processing processing_agent payment_agent financial_agent money_transfer_agent`);
G({ jobScam: 0.5 }, `hiring vacancy vacancies recruit* commission* telecommut* remote_work freelanc*`);

// ---- parcel: fake delivery notices ("your parcel is held, pay the customs fee").
G({ parcel: 1 }, `parcel* undeliver* redeliver* redelivery delivery_failed failed_delivery delivery_attempt
  attempted_delivery customs_fee customs_fees shipping_fee delivery_fee postage_fee held_at on_hold_at
  your_package your_parcel your_shipment your_delivery address_incomplete incomplete_address reschedule_delivery
  tracking_number track_your schedule_delivery confirm_your_address courier`);
G({ parcel: 0.5 }, `package packages shipment* delivery deliveries dispatch* warehouse tracking postage`);

// ---- seo: website, ranking and mailing-list services sold by cold mail.
G({ seo: 1 }, `seo backlink* search_engine search_engines first_page page_one top_of_google google_ranking
  rankings ranking website_design web_design redesign_your your_website your_site lead_generation more_traffic
  website_traffic targeted_traffic bulk_email email_marketing mailing_lists email_list email_lists email_addresses
  million_emails emails_list promote_your boost_your grow_your more_customers new_customers sales_leads`);
G({ seo: 0.5 }, `marketing`);

// ---- renewal: fake invoices and tech-support ("your subscription renewed for $399, call to cancel").
G({ renewal: 1 }, `auto-renew auto_renew autorenew renewed renewal renew renewing subscription_renewed has_been_renewed
  charged deducted debited cancel_your cancel_the cancellation
  refund_department billing_department support_team tech_support technical_support customer_support
  virus_detected your_computer_is your_pc_is geek_squad call_us call_now helpline toll_free`);

// ---- debt: loans, credit repair, debt relief.
G({ debt: 0.4 }, `debt debts`);
G({ debt: 1 }, `debt_relief debt_consolidation consolidate_your credit_score credit_repair fico payday cash_advance loan_approved instant_loan
  guaranteed_loan no_credit_check low_interest interest_rate bankruptcy foreclos* settle_your irs_debt tax_debt
  student_loan student_loans forgiveness lower_your_payment monthly_payment emprestimo emprestimos credito_aprovado`);

// ---- health: miracle health products outside the pharmacy pitch.
G({ health: 1 }, `fat_burn* burn_fat diet_pill diet_pills keto detox* cleanse antiaging anti-aging wrinkle*
  skin_care hair_loss hairloss baldness miracle_cure cure cures supplement supplements testosterone garcinia acai collagen
  blood_sugar diabetes_cure lose_pounds lose_weight weight_loss belly_fat pounds_in inches_off natural_remedy
  doctors_hate one_weird weird_trick`);
G({ health: 0.4 }, `diet dieting lbs fat`);

// ---- more pharmacy, adult, prize, urgency and pitch words
G({ pharma: 1 }, `~viagra cialis* ~levitra ~xanax ~valium ~vicodin ~tramadol ~phentermine ~pharmac kamagra
  priligy dapoxetine clomid accutane zithromax amoxicillin lasix synthroid nexium celebrex
  modafinil provigil ultram fioricet carisoprodol`);
G({ adult: 1 }, `~porn xxx camgirl* camgirls sexcam livecam live_cam webcams escort* hookups horny_girls
  local_girls hot_girls naughty naughtiest slutty threesome`);
G({ prize: 1 }, `gift_card giftcard gift_cards itunes_card google_play_card steam_card amazon_card walmart_card
  survey_reward rewards_program loyalty_reward points_expire your_points redeem* cashback cash_back
  you_have_been_chosen chosen_winner lucky_winner lucky_draw prize_money winning_number winning_numbers
  claim_your_prize claim_your_reward congratulations_you you_are_eligible eligible_for free_iphone free_gift
  premiado premiada contemplado contemplada ganhou_um foi_premiado foi_sorteado`);
G({ prize: 0.4 }, `survey surveys questionnaire iphone ipad macbook ps5 playstation`);
G({ urgent: 1 }, `act_fast act_quickly don't_miss dont_miss before_it's limited_offer offer_expires expires_today
  expires_soon expiring only_today final_reminder last_reminder time_sensitive respond_immediately without_delay
  final_warning last_warning immediate_action action_required required_action urgente_resposta
  suspended suspend suspensa suspenso bloqueio`);
G({ market: 1 }, `limited_stock while_supplies sign_up_today join_now join_free free_membership members_only vip
  exclusive_access lowest_prices factory_direct wholesale_prices order_online buy_online secure_checkout
  add_to_cart checkout shop_online visit_our_website bestseller best_seller top_rated customer_reviews
  as_seen_on satisfaction money-back no_risk risk-free call_toll free_quote free_consultation free_estimate`);
G({ cta: 1 }, `tap follow_this click_the_link tap_the_link use_the_link follow_the_link visit_the_link
  open_the_attachment download_the_attachment click_on click_this click_link tap_here press_here clique_aqui
  clica toque acesse_o aceda_ao atualize atualizar regularize`);
G({ account: 1 }, `mfa 2fa two-factor authenticat* security_code verification_code one-time_password
  restricted* unusual_sign-in sign-in new_device login_attempt failed_login billing_information
  payment_information card_details update_payment payment_failed payment_declined expired_card
  social_security tax_refund tax_return cadastro dados_pessoais seus_dados`);
G({ account: 0.4 }, `cartao credito debito`);
G({ smsspam: 1 }, `freemsg free_msg rply reply_yes reply_win txt_win text_win stop2end stop2stop 2optout
  std_txt_rate msg_rate`);

// ---- a stranger introducing themself: people who know the reader do not say who they are. Counts as salut (a
// message from someone with no prior contact), which the chat neurons read as "not a conversation".
G({ salut: 1 }, `introduce_myself let_me_introduce permit_me_to allow_me_to my_name_is i_am_mr i_am_mrs i_am_miss
  i_am_dr i_am_barrister i_am_the_manager you_do_not_know_me you_don't_know_me we_have_not_met we_have_never_met
  previous_correspondence come_to_you_as as_a_surprise may_come_as might_come_as this_may_surprise
  to_whom_it_may_concern whom_it_may_concern i_got_your_contact got_your_contact your_contact_from
  eu_sou_o eu_sou_a meu_nome_e o_meu_nome_e`);
// ---- the religious register of the advance-fee letter ("Dearest in the Lord", "God fearing")
G({ advfee: 0.4 }, `dearest beloved in_christ in_the_lord god_fearing lord's the_lord almighty`);

// ---- shared context: friends refer to what they already agreed ("that money", "as we agreed", "what I asked you").
// A stranger cannot presuppose a prior agreement, so this is ham evidence even inside a request for money.
G({ shared: 1 }, `aquele_valor aquele_dinheiro aquele_taco aquela_massa aquilo_que como_combinado como_combinamos
  combinado combinamos que_te_pedi te_pedi o_que_te_pedi como_falamos como_falei ja_te_disse
  as_we_agreed as_agreed like_i_said what_i_asked the_money_i i_told_you we_talked_about as_promised`);
// ---- bank and mobile-money notices in Portuguese: one-time codes, recharges, terms, anti-fraud advice.
G({ receipt: 1 }, `use_o_codigo codigo_de_confirmacao em_caso_de_duvida confirmar_a_transaccao recarga recargas
  contador valor_debitado gratos_pela_preferencia pela_preferencia serial iva kwh credelec autorizacao
  transferencia_domestica`);
G({ solicited: 1 }, `termos_e_condicoes entram_em_vigor em_vigor para_evitar verifica_sempre nunca_partilhe
  nunca_partilhes nao_partilhe nao_partilhes never_share do_not_share will_never_ask we_will_never`);

// ---- topic: the subject matter of ordinary writing - science, nature, history, geography, the arts, sport,
// programming, education, cooking, travel. Spam has a handful of subjects; people and publications write about
// everything. Weak evidence on its own, and counted only in prose (see the hamInSalad unit), because word-salad spam
// pours in exactly such words.
G({ topic: 1 }, `science scientist* physics chemistry biology biolog* geology astronom* telescope planet* galaxy
  universe orbit* nasa rocket* satellite* molecule* atom* electron* proton* quantum gravity energy_level evolution
  species fossil* dinosaur* genome gene genes dna protein* cell cells bacteria virus_strain organism* ecology
  ecosystem* climate ocean* glacier* volcano* earthquake* hurricane* tornado* rainforest desert* mountain* river*
  lake lakes island* forest* wildlife bird birds insect* butterfly whale* dolphin* shark* elephant* tiger* lion
  lions wolf wolves bear bears horse horses farm farming harvest crop crops soil seed seeds plant_species botany
  flower* tree trees leaf leaves
  history historian* ancient medieval empire* emperor* kingdom* dynasty century centuries war_of revolution*
  civilization* archaeolog* museum* monument* castle* cathedral* temple* pyramid* roman greek egyptian viking*
  colonial independence constitution democracy parliament_of monarchy pope bible_study philosophy philosoph*
  geography continent* europe european africa african asia asian america american australia canada canadian
  brazil brazilian portugal portuguese mozambique spain spanish france french germany german italy italian
  japan japanese china chinese india indian russia russian mexico mexican england english scotland ireland
  london paris berlin madrid lisbon maputo beira rome tokyo beijing delhi sydney toronto chicago boston texas
  california florida york washington_dc
  art artist* painting* painter* sculpture* gallery galleries poem* poet* poetry literature novel* author* writer*
  fiction chapter* theatre theater opera ballet orchestra symphony jazz rock_band album* lyrics drummer singer*
  actor* actress director* episode* season series documentary cinema comic* manga anime cartoon* photography
  sport sports athlete* olympic* championship* tournament* league* stadium* coach* player* score* scored goal goals
  referee striker* midfielder* cricket rugby baseball hockey marathon cycling_race formula_one boxing wrestling
  chess puzzle* crossword* quiz quizzes trivia hobby hobbies knitting gardening woodwork*
  programming programmer* algorithm* database* query queries recursion compiler interpreter* framework* runtime
  repository* refactor* testing tests unittest exception* stacktrace typescript golang rustlang haskell lisp
  matlab latex fortran cobol pascal assembly kernel_module filesystem* partition* bootloader grub firmware
  education student_union classroom* curriculum lesson* lessons textbook* tutorial_on worksheet* exam_results
  graduation graduate* undergraduate* scholarship_application alumni school_board kindergarten
  cooking recipe_for ingredient* teaspoon* tablespoon* oven bake_at simmer boil* fry frying roast* grill*
  vegetable* onion* garlic tomato* potato* carrot* pepper* spice* herb* flour butter sugar_and salt
  travel tourist* tourism hotel_room hostel* itinerary museum_visit passport visa_application luggage suitcase
  sightseeing backpack* journey* voyage*`);

// ---- the genres of 2000s bulk mail, beyond the core words for them
G({ invest: 1 }, `will_explode get_in_now undiscovered stock_gem huge_news expected_to strong_buy_recommendation
  shares_outstanding market_cap 52_week trading_alert investor_report street_pulse momentum_play
  could_double before_the_market small_cap smallcap microcap big_announcement press_release_expected
  target_price short_term_target projected`);
G({ software: 1 }, `logo_design custom_logo corporate_identity your_logo business_cards stationery_design
  every_type_of_software software_at all_software cheap_software software_for_less retail_price oem_version
  full_version download_now_for cablefilter cable_filter descrambler pay-per-view ppv satellite_tv dish_network`);
G({ health: 1 }, `cholesterol hdl ldl recapture_your your_youth feel_younger look_younger anti-wrinkle hgh_spray
  human_growth growth_hormone antidepressant* prescribed_medication without_a_prescription natural_alternative
  herbal_supplement clinically_proven doctor_recommended`);
G({ adult: 1 }, `mature_lady cheating_wives cheating_on lonely_housewives housewives get_wild orifice office_extravaganza
  barely_legal amateur_video hot_video sex_video wild_party`);
G({ market: 1 }, `this_is_not_spam not_spam you_received_this you_have_received_this at_one_time ffa entered_the
  opt-in opted_in optin your_email_address_was address_was_obtained purchased_list removal_instructions
  to_be_removed remove_in_subject remove_in_the_subject reply_with_remove`);

// ---- completed orders and account events (English): a receipt reports, a phish demands
G({ receipt: 1 }, `has_been_delivered was_delivered been_delivered has_been_shipped has_shipped have_shipped
  has_been_created has_been_processed been_processed your_order_has order_has_shipped order_confirmation_number
  thank_you_for_your_order thanks_for_your_order thank_you_for_shopping thanks_for_shopping you_purchased
  items_you_purchased you_recently_purchased your_recent_order estimated_delivery delivery_date
  you_have_set_up you_set_up`);
G({ receipt: 0.5 }, `your_order order_number`);
// ---- mail sent as a web page, and list digests
G({ solicited: 0.7 }, `as_a_web_page view_this_email view_this_message with_images copy_and_paste into_your_browser
  web_version email_preferences update_your_preferences your_preferences recommends new_recommendations
  recommendations_for_you based_on_items early_bird register_by weekly_news this_week's highlights`);
G({ list: 1 }, `new_messages_in most_active_topics active_topics topics_abridged abridged digest_vol mailing_list_submissions
  list_submissions send_submissions to_subscribe_or_unsubscribe today's_topics pgp_signed begin_pgp`);

// ---- Portuguese chat (Mozambique and Portugal): greetings, texting forms, everyday life, feelings, plans
G({ personal: 1 }, `ola  eish epa pah   fixe bue bwe yah ya sim_sim tas estas tou ta_bem tudo_bem tudo_bom
  bom_dia boa_tarde boa_noite como_estas como_esta como_vai td_bem tb tbm tambem vc vcs pq porque    qdo
  qnd qto qnt msm mesmo mt mto muito mta obg obrigado obrigada obgd valeu beijo beijos bjs bjo abraco abracos
  saudade saudades amigo amiga amigos mano mana brada broh bro dama gaja gajo puto miuda miudo kota cota
  desculpa desculpe perdao calma fica_bem fica_bem_ai ate_logo ate_amanha ate_ja xau tchau chau
  hehe kkk kkkk rsrs lol hahaha haha`);
G({ everyday: 1 }, `casa quarto cozinha sala carro chapa taxi mercado bazar loja_de padaria pao leite arroz feijao
  galinha frango peixe carne batata mandioca matapa xima caril cerveja refresco agua cafe cha comer comi
  comeste beber dormir dormi acordar acordei banho trabalhar trabalho escola aula aulas prova exame teste
  professor professora universidade faculdade igreja culto missa praia jogo futebol filme novela televisao
  musica festa aniversario casamento funeral velorio hospital doente medico remedio chuva sol calor frio
  vento cidade bairro rua estrada viagem viajar machimbombo comboio aviao fim_de_semana feriado ferias`);
G({ family: 1 }, `mamã mama mamae papa papai maezinha paizinho avozinha titio titia cunhado cunhada sogra sogro
  enteado afilhado afilhada padrinho madrinha neto neta bebe crianca criancas filhos familia`);
G({ emotion: 1 }, `contente feliz alegre triste chateado chateada zangado zangada preocupado preocupada cansado
  cansada doente com_saudades estou_bem estou_mal adoro amo gosto_de odeio medo nervoso nervosa orgulho`);
G({ plan: 1 }, `vamos bora vou_ai ja_vou ja_estou estou_a_chegar estou_a_caminho a_caminho chego_ja ja_chego
  passo_ai passa_ca vem_ca vens_ca encontramos nos_vemos ate_logo logo_mais mais_logo amanha_cedo
  depois_falamos falamos_depois liga_me ligo_te ligo_depois`);
G({ timeRef: 1 }, `logo cedo tarde ontem_a_noite hoje_a_noite esta_manha esta_tarde esta_noite semana_passada
  proxima_semana fim_de_semana mes_passado proximo_mes`);
// ---- Spanish phishing and advance-fee wording (the same scams, another language)
G({ account: 1 }, `cuenta contrasena clave verificar verifique verificacion confirmar confirme actualizar actualice
  suspendida suspendido bloqueada bloqueado acceso_no_autorizado actividad_sospechosa su_cuenta tu_cuenta
  datos_personales tarjeta tarjeta_de_credito iniciar_sesion inicie_sesion`);
G({ advfee: 1 }, `herencia heredero difunto fallecido beneficiario fondos_depositados transferencia_de_fondos
  cuestion_de_confianza estrictamente_confidencial propuesta_de_negocio socio_extranjero abogado_personal`);
G({ prize: 1 }, `ganador ganadora premio_mayor ha_ganado usted_ha_ganado felicidades loteria sorteo_internacional`);
G({ urgent: 1 }, `urgente inmediatamente de_inmediato dentro_de_24 dentro_de_48 ultimo_aviso`);
// ---- English texting forms
G({ personal: 1 }, `u2 2u 2day 2nite 2moro 2morrow tmrw tomoz l8r l8 gr8 m8 b4 cu cya ur_welcome np nvm ikr smh
  tho thru wanna gonna gotta lemme gimme kinda sorta dunno ain't y'all ya'll aight ight bruh fam dude mate
  mates cheers innit mum's nah nope yep yup aye oi hiya heya howdy wassup sup ttyl hbu wbu wyd hru omw
  xo xx xxx_kisses love_ya luv_u love_you miss_you miss_ya see_ya talk_soon`);
G({ everyday: 1 }, `uni lecture_hall dorm roommate flatmate housemate commute bus_stop tube metro subway petrol takeaway
  takeout chippy curry kebab noodles sushi burger burgers fries chips crisps biscuit biscuits tea_time
  cuppa telly footy match_day pub_quiz nightout night_out clubbing gig gigs festival festivals`);

// ---- more of what technical and news writing is about
G({ tech: 1 }, `open_source opensource proprietary distro distros distribution_of kde gnome xfce xorg wayland systemd
  grub2 initrd ext4 btrfs zfs lvm raid nfs cifs tcp udp ipv4 ipv6 dhcp nat iptables ldap kerberos ssl tls
  openssl certificate_chain gpg pgp checksum sha1 sha256 md5 tarball rpmbuild makefile autoconf automake
  cmake gdb valgrind strace segfaults coredump backtrace regex regexp unicode utf-8 ascii locale timezone
  utc cron_job crontab rsync scp sftp ssh_key ssh_keys sudo chmod chown symlink mountpoint fstab inode
  vm vms virtualbox vmware qemu kvm xen hypervisor container_image dockerfile yaml_file json_file xml_file
  csv parser parsing tokenizer lexer grammar_file ast bytecode jvm jit garbage_collector heap mutex
  semaphore deadlock race_condition callback closure lambda iterator generator decorator namespace
  inheritance polymorphism interface_for abstract_class subclass superclass constructor destructor
  pointer pointers malloc segmentation buffer_overflow null_pointer off-by-one benchmark benchmarks
  profiler profiling latency_of throughput scalability load_balancer reverse_proxy nginx haproxy squid
  postfix_config exim dovecot courier_imap spamd razor pyzor dcc bayes_db sa-learn sa-update ruleset
  rulesets mailing-list_archives patch_attached attached_patch proposed_patch bug_report bug_reports
  regression_test test_suite testsuite build_system nightly_build release_notes changelog_entry`);
G({ news: 1 }, `reportedly allegedly sources_said officials_say the_ministry the_government opposition_party
  ruling_party coalition referendum ballot polls polling voter voters turnout campaign_trail candidate
  candidates incumbent constituency senator senators congressman congresswoman governor mayor
  supreme_court high_court verdict sentenced convicted acquitted indicted lawsuit plaintiff defendant
  prosecutor prosecutors testimony hearing hearings inquiry commission_report ceasefire treaty sanctions
  diplomat_said embassy refugees_fled humanitarian aid_agencies death_toll casualties wounded
  stock_market_fell shares_fell shares_rose index_rose gdp recession central_bank interest_rates_rose
  quarterly_results earnings_report chief_executive ceo_said analysts_expect`);

// ---- e-mail-account phishing, the forms it took after 2010: quarantine notices, undelivered mail, shared files
G({ accountMail: 1 }, `email_quarantine your_quarantine quarantined messages_could_not could_not_be_delivered
  not_be_delivered undelivered_messages pending_messages pending_incoming release_messages release_/
  shutdown_your request_to_shutdown deactivation_request deactivate_your email_deactivation
  storage_is_full mailbox_is_full mailbox_upgrade upgrade_your_mailbox mailbox_storage email_storage
  you_have_received_a_file received_a_file shared_a_file shared_a_document shared_with_you has_shared_a
  view_shared view_document sign_in_to_view login_to_view to_view_the_document secure_document
  encrypted_message secure_message password_expires password_will_expire keep_same_password
  keep_your_password retain_your_password validate_your_email verify_your_email confirm_your_email
  email_verification account_verification server_message admin_notice it_helpdesk it_department
  webmail_administrator system_administrator mail_administrator email_administrator`);
G({ accountMail: 1 }, `atualizacao_do_outlook caro_funcionario sua_caixa caixa_de_correio caixa_postal
  su_buzon buzon_de_correo correo_electronico_sera`);

// ---- UK premium-rate SMS spam: its fixed phrasing (prize calls, chat lines, ringtone clubs)
G({ smsspam: 1 }, `await_collection awaiting_collection to_claim_call to_claim_txt valid_12hrs valid_for_12hrs
  sae cost_150p 150ppm 1.50_per polys wap_link next_txt your_mobile_number mobile_number_has
  has_been_selected number_has_been your_number_has bonus_caller caller_prize guaranteed_cash
  guaranteed_prize final_attempt final_attempt_to attempt_to_contact contact_u un-redeemed unredeemed
  bonus_points statement_for_your weekly_comp weekly_competition entry_into into_our
  delivery_within_28 within_28_days 28_days chat_line chatline dating_service sexy_chat live_chat_for
  ringtone_club tone_club music_club opt_out_reply reply_stop_to send_stop_to txt_stop_to text_stop_to
  per_week_until until_you_send 4u 2u_now call_from_a_landline from_a_landline
  identifier_code expires_on prize_guaranteed holiday_or holiday_awaits`);

// ---- relation words: read by the attention heads, little or no weight of their own
G({ deixis: 1 }, `here below link links button banner this_link the_link following_link aqui abaixo`);
G({ obligation: 1 }, `must required require* need_to have_to has_to mandatory obliged failure_to fail_to otherwise
  or_else will_be_forced obrigatorio deve`);
G({ timeUnit: 1 }, `hours hrs hour hr days day minutes mins minute weeks week horas dias`);
G({ price: 1 }, `only just per off each price* priced cost costs usd now apenas so_por por_apenas`);

// ---- everyday: life between people - food, home, family, places, weather, pastimes. Spam talks about these rarely;
// mail between people who know each other talks about them all the time.
G({ everyday: 1 }, `breakfast brunch coffee tea beer beers wine pizza cake cakes cookie* bread cheese chicken soup
  sandwich* kitchen cook cooked cooking bake* baking recipe* restaurant* cafe pub   birthday* wedding*
  christmas xmas easter thanksgiving holiday* vacation weekend* trip trips flight* airport train bus car cars drive
  driving drove traffic parking garden house apartment flat room bedroom sofa couch tv movie movies film films music
  song songs concert* band guitar piano game games football soccer match basketball tennis golf gym running walk
  walking swim swimming beach park church school homework teacher exam exams grade grades kids kid children child
  baby dog dogs cat cats pet pets puppy weather rain raining snow snowing sunny cold warm windy storm city town
  village street road neighbour* neighbor* sugar chocolate candy fruit apple* banana* rice pasta salad dessert
  shower laundry dishes groceries grocery supermarket mall haircut doctor dentist hospital sick flu fever cough
  headache picnic camping hiking fishing bike cycling ride   novel   photo photos picture
  pictures selfie camera beach_house road_trip
  comida almoco pequeno_almoco cafe pao escola praia chuva frio calor cidade rua carro chapa machimbombo festa
  aniversario casamento igreja futebol jogo filme musica cerveja frango arroz peixe matapa xima`);
// ---- family: the people in a family. Strong ham evidence in a chat, but the advance-fee story names relatives too
// ("my late husband"), so it is read together with that story by the MLP.
G({ family: 1 }, `mum mom mommy mama dad daddy papa son daughter husband wife hubby brother sister bro sis uncle aunt
  auntie cousin grandma grandpa granny grandad nan nana grandson granddaughter nephew niece in-laws inlaws
  family boyfriend girlfriend fiance fiancee kids
  mae pai filho filha marido esposa mulher irmao irma tio tia avo avó primo prima sobrinho sobrinha mano mana`);
// ---- emotion: how the writer feels.
G({ emotion: 1 }, `happy sad tired exhausted bored excited worried angry upset miss missed missing glad scared
  nervous proud fun funny cute lovely beautiful awesome cool crazy weird annoying annoyed hate hated loved
         hurt cry crying laugh laughed laughing smile happiness
  feliz triste cansado cansada saudade saudades chateado chateada contente gosto adoro`);
// ---- firstPerson: the writer speaking of themself.
G({ firstPerson: 1 }, `i me my mine myself we im i'm i've i'll i'd we're we'll we've we'd
  ive eu meu minha meus minhas estou tenho fui vou`);
// ---- past: things that happened (irregular past tense; "-ed" is an affix filter below)
G({ past: 1 }, `went came saw got had did made said told took gave found thought felt left bought ate drank slept met
  heard knew ran wrote brought caught taught fought sat stood lost forgot spent paid
  fui foi fomos chegou cheguei comprei falei liguei vi comi dormi sai saiu`);
// ---- plan: arrangements between people ("see you at 7", "I'll pick you up")
G({ plan: 1 }, `gonna  later  tonight tomorrow tmr tmrw catch_up meet_up pick_up pick_you come_over coming
  arrive arriving leave leaving heading on_my_way otw be_there be_back get_home see_you see_ya cya ttyl talk_later
  call_you_later call_me_later text_me let_me_know lets let's 
  logo depois amanha vou_ai vens vem_ca chego chegar sair saio encontrar`);
// ---- timeRef: a specific time the writer refers to
G({ timeRef: 1 }, `yesterday today tonight tomorrow morning afternoon evening night monday tuesday wednesday thursday
  friday saturday sunday january february march april june july august september october november december
  last_week next_week last_night this_morning this_weekend this_evening last_year next_year
  ontem hoje amanha manha tarde noite segunda terca quarta quinta sexta sabado domingo`);
// ---- more work, tech and academic talk (ham)
G({ work: 1 }, `coworker* co-worker boss shift rota timesheet* onboarding quarterly kpi kpis okr okrs roadmap
  milestone* deliverable* stakeholder* sync standup stand-up retro sprint backlog ticket* jira slack zoom
  calendar invite invitation reschedule postpone* minutes_of action_items follow_up follow-up headcount
  payroll_team hr_team the_client our_client's colleague interns intern contractor* vendor* procurement
  trabalho reuniao relatorio chefe colega escritorio`);
G({ tech: 1 }, `github gitlab pull_request merge rebase docker kubernetes npm pip css json yaml
  endpoint* backend frontend deploy* unit_test laptop keyboard browser firefox safari router wifi
  bandwidth latency ssh ftp vpn localhost syntax compiler cron daemon ubuntu centos
  computador portatil telemovel`);
G({ academic: 1 }, `assignment* coursework semester tutor campus syllabus lecturer* faculty dean grant* postdoc
  colleagues_at preprint reviewers manuscript* citation*`);
G({ news: 1 }, `according_to officials_said spokesperson government's ministry_of prime_minister police_said
  investigation investigators lawmakers legislation economy economic inflation unemployment`);
G({ personal: 1 }, `xx xoxo hugs kisses babe hun hon bday bff omg lmao rofl tbh idk imo btw brb gtg ty thx
  thnx tks obg obgd bjs bjss beijinhos abraco abracos`);

// ---- anywhere-in-a-word stems for words that hide inside others ("cheapviagra", "unsubscribing")
G({ unsub: 1 }, `~unsubscrib`);
G({ replica: 1 }, `~rolex`);

export function lexicon() { return LEX; }

// ---------------------------------------------------------------- affix filters (fire alongside the winning stem)
// [suffix, minimum word length, row]
export const AFFIXES = [
  ['ed', 6, { past: 0.5 }],    // "-ed": English past tense ("talked", "finished"); half weight, it also marks adjectives
];

// ---------------------------------------------------------------- letter-pair layer
// How a word can be pronounced in English or Portuguese, written as two tables over letter pairs:
//   RARE: pairs that (almost) never occur inside a word of either language - "q" not followed by "u", "j" or "x" or
//         "z" or "v" followed by most consonants, and a few others. Weight 1 each.
//   ONSET: the consonant pairs a word may start with ("bl", "str" -> "st", "ch", "qu"...). A word that starts with two
//         consonants not in this table cannot be read aloud ("dtowa", "lbd", "nsxt").
// A word the stem filters do not know, with at least one vowel and 4+ letters, gets an "unpron" unit:
// clamp(rare pairs + onset violation - 0.5). Read by the gibberish neuron.
const RARE = new Set();
const CONS = 'bcdfghjklmnpqrstvwxz';
for (const c of 'abcdefghijklmnopqrstvwxyz') if (c !== 'u') RARE.add('q' + c);
for (const c of 'bcdfgjklmnpqrstvwxz') RARE.add('j' + c);
for (const c of 'bdfgjkmnqrvwz') RARE.add('x' + c);
for (const c of 'bcdfgjkmnpqrstvwx') RARE.add('z' + c);
for (const c of 'bcdfghjkmnpqtwxz') RARE.add('v' + c);
for (const p of `bk bq bx bz cb cf cg cj cp cv cw cx dx fj fq fv fx fz gq gv gx gz hx hz kq kx kz lx mq mx mz pq pv px
  pz sx sz tq tx wq wx wz yq yj fk fg kg kd kf kp kt kv dk tk pk gk bg bf bp gb gp gd gf pg pd pf pb db dg dp cd cn cm`.trim().split(/\s+/)) RARE.add(p);
export const LETTER_RARE = [...RARE].sort();
export const LETTER_ONSET = `bl br ch cl cr dr dw fl fr gh gl gn gr kl kn kr ph pl pr ps qu rh sc sh sk sl sm sn sp sq st
  sv sw th tr ts tw wh wr zh`.trim().split(/\s+/);

export const RATIOS = [
  ...BASE_RATIOS,
  // share of tokens that are words (not numbers, symbols, links). Word salad is nearly all words; code, tables,
  // forecasts and logs are not. Read only by the prose unit.
  { out: 'alphaRatio', num: '@alpha', den: 'tokens', minWords: 15, empty: 0.8 },
  // share of words that cannot be read aloud (letter-pair layer)
  { out: 'unpronRatio', num: 'unpron', minWords: 8, empty: 0 },
];

// ---------------------------------------------------------------- shapes
export const SHAPES = {
  ...BASE_SHAPES,
  'p?': { question: 1 },          // a question mark
  clock: { timeRef: 1, clock: 1 }, // "7pm", "10am": a time of day
  unpron: { unpron: 1 },           // a word that cannot be read aloud (letter-pair layer)
};

// URL features: the core ones, plus a page planted in a site's asset or system folder, more phishing-kit words, more
// free site builders and shorteners. Each is a fact about how phishing pages are hosted, not a list of bad sites.
export const URL = {
  ...BASE_URL,
  rows: { ...BASE_URL.rows, assetDir: { urlBad: 0.9 }, hiddenDir: { urlBad: 1.2 }, knownSite: { urlKnown: 1 } },
  sets: {
    ...BASE_URL.sets,
    keyword: [...BASE_URL.sets.keyword, 'webapps', 'websrc', 'owa', 'outl', 'oauth', 'auth', 'token', 'sso', 'helpdesk',
      'mailbox', 'webmail', 'liker', 'followers', 'claim', 'reward', 'bonus', 'gift', 'prize', 'unlock', 'suspend',
      'restore', 'appeal', 'enquete', 'cliente', 'client', 'customer', 'service', 'support'],
    freeHost: [...BASE_URL.sets.freeHost, 'creatorlink.net', 'codesandbox.io', 'xf.cz', 'filetransfer.io', 'weebly.com',
      'wordpress.com', 'tilda.ws', 'jimdosite.com', 'mystrikingly.com', 'site123.me', 'webnode.page', 'carrd.co',
      'glitch.me', 'pages.dev', 'workers.dev', 'r2.dev', 'ipfs.io', 'dweb.link', 'firebasestorage.googleapis.com',
      'storage.googleapis.com', 'blob.core.windows.net', 'formstack.com', 'jotform.com', 'typeform.com', 'forms.gle',
      'sway.office.com', 'linktr.ee', 'notion.site'],
    shortener: [...BASE_URL.sets.shortener, 'hyperurl.co', 'rb.gy', 's.id', 'shorte.st', 'adf.ly', 'bitly.com', 't.ly',
      'v.gd', 'qr.co', 'lnkd.in', 'buff.ly', 'urlz.fr', 'clck.ru', 'u.to', 'x.co', 'tny.im'],
    // well-known content sites: a link whose registered domain is one of these is the real site. Platforms where anyone
    // can publish a page (Google, Microsoft, Dropbox, blog and site builders, GitHub Pages) are deliberately not here.
    knownSite: `wikipedia.org wikimedia.org wiktionary.org britannica.com dictionary.com merriam-webster.com
      thesaurus.com imdb.com archive.org github.com gitlab.com bitbucket.org stackoverflow.com stackexchange.com
      superuser.com serverfault.com python.org perl.org cpan.org apache.org mozilla.org gnu.org kernel.org
      debian.org ubuntu.com fedoraproject.org redhat.com suse.com opensuse.org freebsd.org sourceforge.net
      npmjs.com pypi.org rubygems.org w3.org w3schools.com php.net mysql.com postgresql.org oracle.com java.com
      ietf.org rfc-editor.org icann.org eff.org fsf.org linux.com lwn.net slashdot.org
      bbc.co.uk bbc.com cnn.com nytimes.com theguardian.com guardian.co.uk washingtonpost.com reuters.com apnews.com
      bloomberg.com forbes.com wsj.com economist.com ft.com npr.org aljazeera.com usatoday.com latimes.com
      huffpost.com huffingtonpost.com politico.com time.com newsweek.com theatlantic.com wired.com techcrunch.com
      theverge.com arstechnica.com engadget.com zdnet.com cnet.com pcworld.com infoworld.com computerworld.com
      nationalgeographic.com scientificamerican.com nature.com sciencemag.org newscientist.com smithsonianmag.com
      howstuffworks.com instructables.com lifehacker.com mashable.com buzzfeed.com vice.com vox.com
      amazon.com amazon.co.uk ebay.com walmart.com target.com bestbuy.com etsy.com ikea.com apple.com netflix.com
      spotify.com paypal.com adobe.com yahoo.com bing.com duckduckgo.com twitter.com facebook.com instagram.com
      linkedin.com reddit.com pinterest.com youtube.com vimeo.com twitch.tv quora.com imgur.com flickr.com
      tripadvisor.com booking.com airbnb.com expedia.com yelp.com craigslist.org indeed.com glassdoor.com
      espn.com nba.com nfl.com mlb.com fifa.com uefa.com skysports.com goal.com
      rottentomatoes.com metacritic.com goodreads.com allrecipes.com foodnetwork.com weather.com accuweather.com
      who.int un.org europa.eu oecd.org worldbank.org imf.org redcross.org unicef.org
      publico.pt rtp.pt sapo.pt observador.pt dn.pt expresso.pt globo.com uol.com.br folha.uol.com.br
      opais.co.mz noticias.sapo.mz clubofmozambique.com`.trim().split(/\s+/),
    assetDir: ['img', 'images', 'image', 'js', 'css', 'includes', 'include', 'files', 'file', 'uploads', 'upload',
      'assets', 'tmp', 'temp', 'cache', 'cgi', 'cgi-bin', 'media', 'libraries', 'library', 'modules', 'components',
      'templates', 'vendor', 'fonts', 'admin', 'wp-includes', 'lib', 'old', 'backup', 'bak', 'new'],
  },
};

// ---------------------------------------------------------------- attention heads
// query and key are weighted sums over dims (a head fires where its query is positive). layer 2 heads read the
// embeddings plus what layer 1 wrote.
const h1 = (name, query, key, out, before, after, slope = 0.3, beta = 4, sink = 2) => ({ name, layer: 1, query, key, out, before, after, beta, slope, sink });
const h2 = (...a) => ({ ...h1(...a), layer: 2 });
export const HEADS = [
  // ---- layer 1: the core heads
  h1('amount', { bignum: 1, number: 1 }, { currency: 1, money: 1 }, 'amount', 2, 2, 0.5),
  h1('ctaLink', { cta: 1 }, { url: 1, phone: 1, shortcode: 1, email: 1 }, 'ctaLink', 0, 10, 0.2),
  h1('youWin', { prize: 1 }, { you: 1 }, 'youWin', 5, 3),
  h1('youAcct', { account: 1 }, { you: 1 }, 'youAcct', 4, 2),
  h1('payReq', { sendVerb: 1 }, { phone: 1, number: 1 }, 'payReq', 2, 12, 0.15),
  h1('dearEmail', { greet: 1 }, { email: 1 }, 'salut', 0, 2, 0.3, 4, 1),
  h1('named', { greet: 1 }, { capName: 1 }, 'named', 0, 2, 0.5, 4, 1),
  // ---- layer 1: more relations
  // "Dear Friend", "Hello Customer": a greeting followed by a word that names no one (the lexicon alone knows only fixed pairs)
  h1('stranger', { greet: 1 }, { addressee: 1 }, 'salut', 0, 2, 0.5, 4, 1),
  // "click here http://...", "the link below": a pointer word whose target is a link or address
  h1('deixisLink', { deixis: 1 }, { url: 1, email: 1, phone: 1 }, 'ctaLink', 2, 8, 0.25),
  // "within 24 hours", "expires in 3 days": pressure with a clock on it
  h1('deadline', { urgent: 1, obligation: 0.5, accountMail: 0.5 }, { timeUnit: 1, number: 0.5 }, 'deadline', 3, 6, 0.3),
  // "only $19.95", "just 49 each": a price
  h1('priceTag', { currency: 1, amount: 0.5 }, { price: 1 }, 'priceTag', 3, 3, 0.4),
  // "70% off", "save 50%"
  h1('discount', { percent: 1 }, { market: 1, price: 1 }, 'discount', 3, 3, 0.4),
  // "PayPal: your account is limited": a brand next to account or pressure talk
  h1('impersonate', { brand: 1 }, { account: 1, accountMail: 1, urgent: 0.7 }, 'impersonate', 4, 8, 0.2),
  // "you must verify", "you are required to"
  h1('pressure', { you: 1 }, { obligation: 1 }, 'pressure', 1, 3, 0.4),
  // "I went", "we talked": the writer narrating what happened
  h1('narrate', { firstPerson: 1 }, { past: 1 }, 'narrate', 0, 3, 0.4),
  // "I'll pick you up", "we're going tonight": the writer making a plan
  h1('arrange', { firstPerson: 1, you: 0.5 }, { plan: 1, timeRef: 0.7 }, 'arrange', 1, 4, 0.4),
  // "send 0.1 BTC to this wallet"
  h1('cryptoPay', { crypto: 1 }, { sendVerb: 1, amount: 1, number: 0.5, random: 1, money: 0.5 }, 'cryptoPay', 6, 8, 0.2),
  // "I recorded you ... pay ... bitcoin"
  h1('extortPay', { extort: 1 }, { crypto: 1, money: 1, deadline: 1, timeUnit: 0.7 }, 'extortPay', 10, 20, 0.08),
  // "earn $500 a day", "paid weekly 3,000"
  h1('jobPay', { jobScam: 1 }, { amount: 1, money: 1, currency: 1, timeUnit: 0.5 }, 'jobPay', 6, 10, 0.2),
  // "your parcel ... pay the 1.99 fee"
  h1('parcelFee', { parcel: 1 }, { currency: 1, amount: 1, price: 0.5, url: 1, account: 0.5 }, 'parcelFee', 6, 12, 0.15),
  // "free pills", "free shipping on all watches": something free on offer
  h1('offerFree', { prize: 1 }, { market: 1, pharma: 1, software: 1, adult: 1, replica: 1, health: 1 }, 'offerFree', 3, 3, 0.4),
  // "how was your day?": a question about someone's life
  h1('chatQ', { question: 1 }, { personal: 1, everyday: 1, family: 1, emotion: 1, you: 0.3 }, 'chatQ', 6, 0, 0.3),
  // "renewed for $399 ... call 1-800": a fake invoice with a number to call
  h1('techScam', { renewal: 1 }, { phone: 1, amount: 1, currency: 0.7 }, 'techScam', 8, 12, 0.15),
  // "tell your mom", "my sister's wedding": family in everyday talk
  h1('familyTalk', { family: 1 }, { everyday: 1, emotion: 1, plan: 1, timeRef: 0.5, firstPerson: 0.5 }, 'familyTalk', 4, 4, 0.4),
  // "unsubscribe: http://...", "to be removed reply ...": a removal footer whose target is a link or address
  h1('unsubLink', { unsub: 1 }, { url: 1, email: 1 }, 'unsub', 2, 8, 0.25),
  // "John Smith wrote:", "jose@... escreveu:": a reply attribution (quoting someone who wrote to you)
  h1('wroteBy', { list: 1 }, { capName: 1, email: 1, named: 1 }, 'quote', 4, 0, 0.4),
  // "PayPal ... http://paypal-secure.example.tk": a brand named next to a bad-looking link
  h1('brandLink', { brand: 1 }, { urlBad: 1 }, 'impersonate', 3, 12, 0.15),
  // "can I come over?", "did we pay the bill?": the writer asking about their own plans
  h1('selfQuestion', { question: 1 }, { firstPerson: 1, plan: 0.5 }, 'chatQ', 6, 0, 0.3),
  // "transfer $500 today", "pay 1.99 now": a sum with pressure next to it
  h1('moneyNow', { amount: 1, currency: 0.5 }, { urgent: 1, obligation: 0.7, timeUnit: 0.5 }, 'deadline', 4, 6, 0.3),
  // "free for 30 days", "trial 14 days": a free offer on a clock
  h1('freeTrial', { prize: 1 }, { timeUnit: 1 }, 'offerFree', 2, 4, 0.4),
  // ---- layer 2 (reads layer 1)
  // an offer whose next step is a link or number: "70% off ... http://", "free pills, order at ..."
  h2('offerLink', { offerFree: 1, priceTag: 1, discount: 1, amount: 0.7, youWin: 1 }, { url: 1, phone: 1, ctaLink: 1, shortcode: 1 }, 'offerLink', 2, 12, 0.15),
  // a threat whose next step is a link or number: "within 24 hours ... verify here http://"
  h2('threatLink', { deadline: 1, pressure: 1, impersonate: 1, extortPay: 1 }, { url: 1, ctaLink: 1, phone: 1, email: 1 }, 'threatLink', 3, 15, 0.1),
  // the advance-fee story told in the first person: "my late husband deposited $10.5 million"
  h2('storyMoney', { narrate: 1, family: 0.5 }, { amount: 1, advfee: 1, bignum: 0.5 }, 'storyMoney', 6, 10, 0.15),
  // a plan with a time: "see you at 7pm", "we're meeting tomorrow"
  h2('meetup', { arrange: 1 }, { clock: 1, timeRef: 1, everyday: 0.5 }, 'meetup', 4, 4, 0.4),
  // a stranger's greeting followed by the story's money: "Dear Friend ... US$ 12.5 million"
  h2('strangerMoney', { salut: 1 }, { amount: 1, storyMoney: 1, bignum: 0.5 }, 'storyMoney', 0, 40, 0.05),
  // a threat followed by a demand to pay: "your account will be closed ... pay the fee"
  h2('threatPay', { deadline: 1, pressure: 1 }, { sendVerb: 1, amount: 1, priceTag: 1 }, 'threatLink', 2, 12, 0.15),
];

// ---------------------------------------------------------------- sentence roles (the core roles, extended)
const CORE_ROLES = Object.fromEntries(BASE_ROLES.map(r => [r.name, r]));
export const ROLES = [
  { ...CORE_ROLES.offer, w: { ...CORE_ROLES.offer.w, crypto: 1.0, gamble: 1.2, jobScam: 1.0, jobPay: 1.2, seo: 1.0, debt: 1.0, health: 1.0,
    offerFree: 1.2, priceTag: 1.0, discount: 1.2 } },
  { ...CORE_ROLES.ask, w: { ...CORE_ROLES.ask.w, pressure: 1.0, deadline: 1.0, impersonate: 1.0, parcelFee: 1.0, cryptoPay: 1.2, techScam: 1.2 } },
  { ...CORE_ROLES.contact, w: { ...CORE_ROLES.contact.w, offerLink: 1.0, threatLink: 1.0 } },
  { ...CORE_ROLES.story, w: { ...CORE_ROLES.story.w, storyMoney: 1.2 } },
  CORE_ROLES.notice,
  { ...CORE_ROLES.chat, w: { ...CORE_ROLES.chat.w, chatQ: 1.0, emotion: 0.8, familyTalk: 1.0, shared: 1.2 } },
  { ...CORE_ROLES.work, w: { ...CORE_ROLES.work.w, topic: 0.4 } },
  { ...CORE_ROLES.news, w: { ...CORE_ROLES.news.w, topic: 0.3 } },
  { name: 'threat', b: -0.3, why: 'something bad will happen unless the reader acts: a recording, a suspension, a deadline',
    w: { extort: 1.2, extortPay: 1.5, deadline: 1.0, pressure: 0.8, impersonate: 0.8, accountMail: 0.6 } },
  { name: 'life', b: -0.4, why: 'the writer\'s own life: what happened, what is planned, family, food, feelings',
    w: { everyday: 0.8, emotion: 0.8, narrate: 1.0, arrange: 1.2, meetup: 1.2, family: 0.6, familyTalk: 1.0, timeRef: 0.3 } },
];
export const ARCS = [
  ...BASE_ARCS,
  { name: 'threatAsk', from: 'threat', to: 'ask', window: 2, same: true },
  { name: 'threatContact', from: 'threat', to: 'contact', window: 3, same: true },
];

// ---------------------------------------------------------------- MLP
// layer 1: concept neurons (the core ones, plus more scams and more ham); layer 2: gates and message shape (the core
// second layer, extended); layer 3: combinations of the two.
const CORE1 = Object.fromEntries(BASE_HIDDEN.map(h => [h.name, h]));
const CORE2 = Object.fromEntries(BASE_HIDDEN2.map(h => [h.name, h]));
export const MLP = [
  [
    ...BASE_HIDDEN.filter(h => !['payRequest', 'phishing', 'urlVerdict'].includes(h.name)),
    { ...CORE1.urlVerdict, w: { ...CORE1.urlVerdict.w, urlKnown: -2.5 },
      why: CORE1.urlVerdict.why + ' (a lone link to a well-known content site is that site; spam quotes such links inside messages, so they vouch only for a lone link)' },
    { ...CORE1.phishing, w: { ...CORE1.phishing.w, named: -1, solicited: -0.6, receipt: -0.8, list: -0.5 },
      why: CORE1.phishing.why + ' (phishers rarely know your name, and a receipt reports rather than demands)' },
    { ...CORE1.payRequest, w: { ...CORE1.payRequest.w, shared: -2 }, why: CORE1.payRequest.why + ' (and it is not money the two already agreed on)' },
    { name: 'cryptoScam', w: { cryptoPay: 1.2, crypto: 0.5, amount: 0.5, invest: 0.3, prize: 0.3, news: -1, tech: -0.7, quote: -0.5, named: -0.5, shared: -0.8, receipt: -0.6 }, b: -1.3,
      why: 'coins plus a payment or a sum: doubling, recovery and investment scams' },
    { name: 'extortion', w: { extort: 0.6, extortPay: 1.5, crypto: 0.5, deadline: 0.5, tech: -0.8, list: -0.5, quote: -0.5, named: -0.5, shared: -0.8, receipt: -0.6 }, b: -1.5,
      why: '"I recorded you" plus payment or a deadline' },
    { name: 'jobOffer', w: { jobScam: 1, jobPay: 1.2, amount: 0.4, money: 0.3, work: -0.4, academic: -0.5, list: -0.5, quote: -0.5, named: -0.5, shared: -0.8, receipt: -0.6 }, b: -1.4,
      why: 'easy work for good pay' },
    { name: 'parcelScam', w: { parcel: 0.7, parcelFee: 1.2, urlBad: 0.5, ctaLink: 0.5, brand: 0.3, tech: -0.8, work: -0.4, energy: -0.8, quote: -0.5, named: -0.5, shared: -0.8, receipt: -0.6 }, b: -1.5,
      why: 'a held parcel and a fee or link' },
    { name: 'seoPitch', w: { seo: 1, market: 0.3, you: 0.2, tech: -0.6, work: -0.4, list: -0.6, quote: -0.5, named: -0.5, shared: -0.8, receipt: -0.6 }, b: -1.3,
      why: 'web and mailing-list services sold cold' },
    { name: 'gambling', w: { gamble: 1, prize: 0.3, offerFree: 0.5, news: -1, quote: -0.5, named: -0.5, shared: -0.8, receipt: -0.6 }, b: -0.9,
      why: 'casinos and betting' },
    { name: 'impersonation', w: { impersonate: 1.2, brand: 0.2, account: 0.3, urlBad: 0.5, threatLink: 0.8, named: -0.5, solicited: -0.8, list: -0.5, tech: -0.4, quote: -0.5, shared: -0.8, receipt: -1.5 }, b: -1.5,
      why: 'a brand talking about your account' },
    { name: 'techSupport', w: { techScam: 1.5, renewal: 0.6, phone: 0.3, receipt: -0.8, tech: -0.4, quote: -0.5, named: -0.5, shared: -0.8 }, b: -1.3,
      why: 'a renewal or a virus warning with a number to call' },
    { name: 'debtPitch', w: { debt: 1, invest: 0.3, offerLink: 0.5, news: -1, work: -0.4, quote: -0.5, named: -0.5, shared: -0.8, receipt: -0.6 }, b: -1.2,
      why: 'loans, credit repair, debt relief' },
    { name: 'healthPitch', w: { health: 1, offerFree: 0.5, offerLink: 0.6, priceTag: 0.4, news: -1, everyday: -0.2, quote: -0.5, named: -0.5, shared: -0.8, receipt: -0.6 }, b: -1.3,
      why: 'miracle health products' },
    { name: 'gibberish', w: { unpronRatio: 12, unpron: 0.3 }, b: -1.2,
      why: 'more than ~10% of the words cannot be read aloud: fragments and strings made to slip past filters' },
    { name: 'offerToAct', w: { offerLink: 1.0, threatLink: 1.0, quote: -0.5, named: -0.5, shared: -0.8, receipt: -0.6 }, b: -0.4,
      why: 'token-level: an offer or a threat followed by a link or number to act on it' },
    { name: 'chatter', w: { personal: 0.5, everyday: 0.4, emotion: 0.5, arrange: 1, chatQ: 1, meetup: 1.2, familyTalk: 0.8, family: 0.2,
      salut: -1.5, advfee: -1, storyMoney: -1, amount: -0.5 }, b: -1.0,
      why: 'people who know each other talking about their lives (a stranger, a sum or the advance-fee story switch it off)' },
    { name: 'prose', clamp: true, w: { funcRatio: 5, alphaRatio: -4 }, b: 3,
      why: '0 only for word salad: few function words AND nearly every token a word. Prose has function words; code, '
        + 'tables and forecasts have few, but are full of numbers and symbols. Short texts count as prose' },
    { name: 'lifeStory', w: { narrate: 1, everyday: 0.4, emotion: 0.3, advfee: -1, amount: -0.8, storyMoney: -2, salut: -1 }, b: -0.9,
      why: 'the writer telling what happened to them, with no sum and no stranger (the advance-fee story has both)' },
  ],
  [
    CORE2.channel,
    { ...CORE2.looseNoChannel, w: { ...CORE2.looseNoChannel.w, offerFree: 0.5, gamble: 0.3 } },
    { ...CORE2.scamArc, w: { ...CORE2.scamArc.w, a_threatAsk: 1, a_threatContact: 1 } },
    { ...CORE2.spamShape, w: { ...CORE2.spamShape.w, s_threat: 1, s_life: -1 } },
    { ...CORE2.hamShape, w: { ...CORE2.hamShape.w, s_life: 1, s_threat: -1 } },
    { name: 'hamInSalad', b: 0, w: { work: 0.5, tech: 0.5, academic: 0.5, news: 0.5, everyday: 0.5, emotion: 0.5, personal: 0.3, list: 0.3, topic: 0.3 }, h: { prose: -20 },
      why: 'ham words in text that is not prose: dictionary words poured into spam to look like ordinary mail. '
        + 'The output gives this back the weight the ham words took away, so ham words count only in sentences a person wrote' },
    { name: 'chatNoChannel', clamp: true, b: 0, w: {}, h: { chatter: 0.6, lifeStory: 0.4, channel: -1 },
      why: 'people talking about their lives with nothing to click or call: a conversation, whatever loose words it has' },
  ],
  [
    { name: 'scamInChat', b: -1, h: { cryptoScam: 1, extortion: 1, impersonation: 1, techSupport: 1, parcelScam: 1, chatter: -0.5 },
      why: 'a scam neuron that fires even though the message chats (romance and recovery scams chat first)' },
  ],
];

// ---------------------------------------------------------------- output
export const OUT = {
  b: BASE_OUT.b,
  w: {
    ...BASE_OUT.w,
    crypto: 0.4, gamble: 0.8, extort: 0.6, jobScam: 0.5, parcel: 0.2, seo: 0.6, renewal: 0.2, brand: 0.1, health: 0.6,
    debt: 0.6, everyday: -0.15, emotion: -0.15, timeRef: -0.05, shared: -1.0, topic: -0.25,
    deadline: 0.6, priceTag: 0.6, discount: 0.6, pressure: 0.4, arrange: -0.2,
    meetup: -0.6, chatQ: -0.5, familyTalk: -0.4, storyMoney: 0.6,
  },
  hidden: {
    ...BASE_OUT.hidden,
    cryptoScam: 1.5, extortion: 2.0, jobOffer: 1.2, parcelScam: 1.5, seoPitch: 1.0, gambling: 1.2, impersonation: 1.2,
    techSupport: 1.5, debtPitch: 1.0, healthPitch: 1.0, offerToAct: 0.6, chatter: -0.6, lifeStory: -0.6,
    chatNoChannel: -0.8, scamInChat: 0.8, hamInSalad: 0.8, gibberish: 1.2,
  },
};
