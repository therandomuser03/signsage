-- SignSage 002: account-scoped, session-based translation history.
--
-- Apply after 001_translation_sessions.sql. Safe to run more than once and
-- safe on databases where translation_messages was created by hand
-- (the table was never owned by a migration).
--
-- Data: NOTHING is deleted, rewritten or invented. Existing rows are valid
-- under the new model: every legacy session holds exactly one sentence
-- message (history used to insert one session + one message per translation
-- call), so legacy entries simply surface as one-sentence sessions. They
-- keep their original timestamps, direction, text, confidence and model.
-- Grouping legacy rows into multi-sentence sessions is impossible -- the
-- old schema had no notion of an interaction -- so we do not pretend.
--
-- Structure after this migration:
--   translation_sessions   one row per translation interaction (a "session")
--     + ended_at           when the interaction/session was closed (WS disconnect / one-shot)
--     + updated_at         bumped on every sentence insert
--     + client_key         optional idempotency key supplied by the client
--                          (a new interaction generates a new key); used to
--                          get-or-create the session so reconnects never fork it
--   translation_messages   one row per completed sentence inside a session
--     + client_key         optional idempotency key per finalized sentence;
--                          unique per session, so retries/duplicates collapse
--   FK messages.session_id -> sessions.id ON DELETE CASCADE
--
-- Ownership stays enforced by RLS (user JWT) plus user_id filters in the
-- repository layer. No service-role key is involved.

-- ---------------------------------------------------------------------------
-- 1) translation_messages: create if missing, then top up missing columns.
-- ---------------------------------------------------------------------------
create table if not exists public.translation_messages (
    id            uuid primary key default gen_random_uuid(),
    session_id    uuid not null,
    sequence      integer not null default 1,
    direction     text not null,
    input_text    text,
    output_text   text not null default '',
    gloss         text,
    confidence    real,
    model_version text not null default '',
    result_type   text not null default 'model',
    created_at    timestamptz not null default now()
);

alter table public.translation_messages add column if not exists session_id uuid;
alter table public.translation_messages add column if not exists sequence integer not null default 1;
alter table public.translation_messages add column if not exists direction text;
alter table public.translation_messages add column if not exists input_text text;
alter table public.translation_messages add column if not exists output_text text not null default '';
alter table public.translation_messages add column if not exists gloss text;
alter table public.translation_messages add column if not exists confidence real;
alter table public.translation_messages add column if not exists model_version text not null default '';
alter table public.translation_messages add column if not exists result_type text not null default 'model';
alter table public.translation_messages add column if not exists created_at timestamptz not null default now();
alter table public.translation_messages add column if not exists client_key text;

-- ---------------------------------------------------------------------------
-- 2) translation_sessions: session lifecycle + idempotency columns.
-- ---------------------------------------------------------------------------
alter table public.translation_sessions add column if not exists ended_at timestamptz;
alter table public.translation_sessions add column if not exists updated_at timestamptz not null default now();
alter table public.translation_sessions add column if not exists client_key text;
-- Sessions are created lazily on the first completed sentence, with that
-- sentence's type/model, so keep a safe default for direct inserts.
alter table public.translation_sessions alter column result_type set default 'model';
alter table public.translation_sessions alter column output_text set default '';
alter table public.translation_sessions alter column model_version set default '';

-- ---------------------------------------------------------------------------
-- 3) Foreign key with cascade, upgraded in place when possible.
-- ---------------------------------------------------------------------------
do $$
declare
    fk_name text;
    fk_del  "char";
begin
    select conname, confdeltype into fk_name, fk_del
    from pg_constraint
    where conrelid = 'public.translation_messages'::regclass
      and confrelid = 'public.translation_sessions'::regclass
      and contype = 'f'
    limit 1;

    if fk_name is null then
        alter table public.translation_messages
            add constraint translation_messages_session_id_fkey
            foreign key (session_id) references public.translation_sessions (id) on delete cascade;
    elsif fk_del <> 'c' then
        -- Replace a plain (non-cascading) FK created by hand so deleting a
        -- session always takes its sentences with it, atomically.
        execute format('alter table public.translation_messages drop constraint %I', fk_name);
        alter table public.translation_messages
            add constraint translation_messages_session_id_fkey
            foreign key (session_id) references public.translation_sessions (id) on delete cascade;
    end if;
end $$;

-- ---------------------------------------------------------------------------
-- 4) Indexes.
-- ---------------------------------------------------------------------------
create index if not exists translation_sessions_user_created_idx
    on public.translation_sessions (user_id, created_at desc);
create index if not exists translation_messages_session_seq_idx
    on public.translation_messages (session_id, sequence);
create unique index if not exists translation_sessions_user_client_key_uniq
    on public.translation_sessions (user_id, client_key) where client_key is not null;
create unique index if not exists translation_messages_session_client_key_uniq
    on public.translation_messages (session_id, client_key) where client_key is not null;

-- ---------------------------------------------------------------------------
-- 5) Row-level security: owner-only access; one message follows its session.
-- ---------------------------------------------------------------------------
-- Sessions gained a lifecycle (ended_at / status / updated_at), so they are
-- no longer strictly immutable. Only the owner may UPDATE, and only the
-- three lifecycle columns -- the translated text stays immutable even for
-- the row owner. The application PATCHes exactly these columns.
do $$
begin
    if not exists (select 1 from pg_policies where schemaname = 'public'
                   and tablename = 'translation_sessions' and policyname = 'own rows: update') then
        create policy "own rows: update" on public.translation_sessions
            for update to authenticated
            using (auth.uid() = user_id)
            with check (auth.uid() = user_id);
    end if;
end $$;

revoke update on public.translation_sessions from authenticated;
grant update (ended_at, updated_at, status) on public.translation_sessions to authenticated;

alter table public.translation_messages enable row level security;

do $$
begin
    if not exists (select 1 from pg_policies where schemaname = 'public'
                   and tablename = 'translation_messages' and policyname = 'own messages: select') then
        create policy "own messages: select" on public.translation_messages
            for select to authenticated
            using (session_id in (select id from public.translation_sessions where user_id = auth.uid()));
    end if;
    if not exists (select 1 from pg_policies where schemaname = 'public'
                   and tablename = 'translation_messages' and policyname = 'own messages: insert') then
        create policy "own messages: insert" on public.translation_messages
            for insert to authenticated
            with check (session_id in (select id from public.translation_sessions where user_id = auth.uid()));
    end if;
    if not exists (select 1 from pg_policies where schemaname = 'public'
                   and tablename = 'translation_messages' and policyname = 'own messages: delete') then
        create policy "own messages: delete" on public.translation_messages
            for delete to authenticated
            using (session_id in (select id from public.translation_sessions where user_id = auth.uid()));
    end if;
end $$;
